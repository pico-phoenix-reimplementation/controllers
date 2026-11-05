#!/usr/bin/env python3
"""
PICO 4 motion controllers -> Linux gamepads (uinput).

The controllers talk to the Nordic station MCU, which streams their packets
into the SLPI sensor "ble_mcu" (sns_ble_mcu). This daemon enables that SSC
sensor (ssc-ble-mcu.py), decodes the 200-byte packets and exposes one
uinput gamepad per hand.

Packet layout (reverse engineered 2026-10-20, AGENTS.md §S):
  0      0xa5 (0x00 = empty slot)
  3      trigger 0..255          6   grip 0..255
  4, 5   stick X / Y (128 = centre)
  7      bits: 0x01 A/X  0x02 B/Y  0x04 menu  0x08 stick click
               0x10 grip click  0x20 home?  0x40 left hand  0x80 trigger click
  8      touch bits: 0x01 A/X  0x02 B/Y  0x04 stick  0x08 thumbrest
                     0x10 trigger
  9      status: 02 (~30 Hz mode), 05 left / 01 right (~500 Hz mode)
  33..44 int16 gyro x/y/z, accel x/y/z      47, 55  u32 timestamps (0 = invalid)

Controller IMU (2026-10-21 guided captures): ICM-42686 at +-32 g (1024 LSB/g)
and +-4000 dps (8.2 LSB/dps). In the ~500 Hz mode (byte 9 = 05 left / 01
right, 2026-10-25) the gyro is in the accel axes as is: a fit of d(gravity)/dt
= g x w over a rotated recording picks (x, y, z) with the right sign and the
datasheet scale for both hands. (The 2026-10-21 (y, -x, z) came from ~30 Hz
packets with byte 9 = 02.) One sample per packet.

  99     CRC-8/MAXIM (reflected poly 0x8c, init 0) of bytes 0..98. ~10% of
         the packets fail it (bit errors on the radio/SLPI path: wrong IMU
         values, flipped hand flag); they are dropped.

With --xr it also feeds the Monado "pico4" driver (monado-pico4/): while the
driver's datagram socket exists, the SLPI gyro/accel are streamed at 500 Hz
and every head IMU sample and controller packet is forwarded to it.
"""
import fcntl
import importlib.util
import os
import socket
import struct
import sys
import time

EV_SYN, EV_KEY, EV_ABS = 0x00, 0x01, 0x03
SYN_REPORT = 0
ABS_X, ABS_Y, ABS_Z, ABS_RZ = 0x00, 0x01, 0x02, 0x05
BTN_SOUTH, BTN_EAST, BTN_TL, BTN_TR, BTN_TL2, BTN_TR2 = 0x130, 0x131, 0x136, 0x137, 0x138, 0x139
BTN_SELECT, BTN_START, BTN_MODE, BTN_THUMBL, BTN_THUMBR = 0x13a, 0x13b, 0x13c, 0x13d, 0x13e
BUS_VIRTUAL = 0x06

UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_SET_ABSBIT = 0x40045567
UI_DEV_SETUP = 0x405C5503
UI_ABS_SETUP = 0x401C5504
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502

XR_SOCKET = os.environ.get("PICO4_SOCKET", "/run/pico4-xr.sock")
XR_IMU_RATE = 500.0
# Accel only corrects the head tilt (the gyro drives the messages): 100 Hz is
# plenty and saves ~400 SSC indications/s of Python work.
XR_ACCEL_RATE = 100.0
DUMP_TRIGGER = "/run/pico4/ctrl-dump"
DUMP_FILE = "/run/pico4/ctrl-dump.txt"
SSC_TICK_NS = 1e9 / 19.2e6      # SSC timestamps are QTimer ticks

# Bits of struct pico4_msg_controller.buttons (pico4_interface.h).
# Raw byte 7 bits (decoded from a labelled capture, 2026-10-26): 0x04 A/X,
# 0x08 B/Y, 0x02 menu (left) / screenshot (right), 0x01 home (right PICO
# button, left programmable button), 0x20 stick click, 0x10 grip click,
# 0x80 trigger click.
XR_BUTTONS = ((0x04, 1 << 0), (0x08, 1 << 1), (0x02, 1 << 2), (0x20, 1 << 3),
              (0x01, 1 << 4), (0x80, 1 << 5), (0x10, 1 << 6))
XR_TOUCHES = ((0x01, 1 << 8), (0x02, 1 << 9), (0x04, 1 << 10), (0x08, 1 << 11),
              (0x10, 1 << 12))

CTRL_ACCEL_LSB_PER_G = 1024.0
CTRL_GYRO_LSB_PER_DPS = 8.2


def _crc8_maxim_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if crc & 1 else crc >> 1
        table.append(crc)
    return bytes(table)


_CRC8_TABLE = _crc8_maxim_table()


def crc8_maxim(data):
    # Table-driven: ~1000 packets/s of 99 bytes; the bit loop alone used
    # about half a CPU core on the headset.
    crc = 0
    t = _CRC8_TABLE
    for byte in data:
        crc = t[crc ^ byte]
    return crc


def packet_valid(raw):
    # Byte 9 is a status field (02 on 2026-10-20, 05/01 left/right later), not
    # a validity flag: the CRC is what rejects corrupted packets.
    return (len(raw) >= 100 and crc8_maxim(raw[:99]) == raw[99]
            and struct.unpack_from("<I", raw, 47)[0] != 0)


def ctrl_imu(raw):
    """Raw gyro (rad/s, bias included: the driver's filter estimates it) and
    accel (m/s^2), controller IMU axes, plus the controller's us counter."""
    g = struct.unpack_from("<3h", raw, 33)
    a = struct.unpack_from("<3h", raw, 39)
    k = 3.141592653589793 / 180.0 / CTRL_GYRO_LSB_PER_DPS
    ka = 9.80665 / CTRL_ACCEL_LSB_PER_G
    return (tuple(x * k for x in g), tuple(x * ka for x in a),
            struct.unpack_from("<I", raw, 47)[0])


# byte 7 bit -> key, per hand (left uses the "L" side codes)
BUTTONS = {
    0x04: (BTN_SOUTH, BTN_SOUTH),     # X / A
    0x08: (BTN_EAST, BTN_EAST),       # Y / B
    0x02: (BTN_SELECT, BTN_START),    # menu (left) / screenshot (right)
    0x20: (BTN_THUMBL, BTN_THUMBR),   # stick click
    0x10: (BTN_TL, BTN_TR),           # grip click
    0x01: (BTN_MODE, BTN_MODE),       # home / programmable
    0x80: (BTN_TL2, BTN_TR2),         # trigger click
}


class Pad:
    def __init__(self, name, left):
        self.left = left
        self.fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_KEY)
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_ABS)
        for codes in BUTTONS.values():
            fcntl.ioctl(self.fd, UI_SET_KEYBIT, codes[0 if left else 1])
        for code, lo, hi in ((ABS_X, 0, 255), (ABS_Y, 0, 255),
                             (ABS_Z, 0, 255), (ABS_RZ, 0, 255)):
            fcntl.ioctl(self.fd, UI_SET_ABSBIT, code)
            fcntl.ioctl(self.fd, UI_ABS_SETUP,
                        struct.pack("<HHiiiiii", code, 0, 128 if code < 2 else 0,
                                    lo, hi, 2 if code < 2 else 0, 4 if code < 2 else 0, 0))
        setup = struct.pack("<HHHH80sI", BUS_VIRTUAL, 0x2d40, 0x0100 + left, 1,
                            name.encode(), 0)
        fcntl.ioctl(self.fd, UI_DEV_SETUP, setup)
        fcntl.ioctl(self.fd, UI_DEV_CREATE)
        self.state = {}

    def emit(self, etype, code, value):
        now = time.time()
        os.write(self.fd, struct.pack("<qqHHi", int(now), int(now % 1 * 1e6),
                                      etype, code, value))

    def update(self, p):
        changed = False
        new = {(EV_ABS, ABS_X): p[5], (EV_ABS, ABS_Y): 255 - p[4],
               (EV_ABS, ABS_Z): p[3], (EV_ABS, ABS_RZ): p[6]}
        for bit, codes in BUTTONS.items():
            new[(EV_KEY, codes[0 if self.left else 1])] = int(bool(p[7] & bit))
        for k, v in new.items():
            if self.state.get(k) != v:
                self.emit(k[0], k[1], v)
                changed = True
        self.state = new
        if changed:
            self.emit(EV_SYN, SYN_REPORT, 0)

    def close(self):
        fcntl.ioctl(self.fd, UI_DEV_DESTROY)
        os.close(self.fd)


class XrBridge:
    """Forwards head IMU and controller state to the Monado pico4 driver."""

    def __init__(self, ssc_mod, ssc):
        self.m = ssc_mod
        self.ssc = ssc
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.gyro = (ssc_mod.lookup(ssc, "gyro") or [None])[0]
        self.accel = (ssc_mod.lookup(ssc, "accel") or [None])[0]
        if self.gyro is None or self.accel is None:
            print("xr: no gyro/accel SUID, head tracking disabled", flush=True)
        self.streaming = False
        self.last_accel = (0.0, 9.81, 0.0)
        self.last_state, self.last_state_t = {}, {}

    def poll(self):
        """Start/stop the IMU streams when the driver socket appears/goes."""
        want = os.path.exists(XR_SOCKET) and self.gyro is not None
        if want == self.streaming:
            return
        for suid in (self.gyro, self.accel):
            if want:
                rate = XR_ACCEL_RATE if suid == self.accel else XR_IMU_RATE
                req = self.m.client_request(suid, self.m.MSGID_STD_SENSOR_CONFIG,
                                            self.m.pb_float(1, rate))
            else:
                req = self.m.client_request(suid, self.m.MSGID_DISABLE_REQ)
            self.ssc.send(req)
        self.streaming = want
        print("xr: head IMU %s" % ("on" if want else "off"), flush=True)

    def send(self, msg):
        if not self.streaming:
            return
        try:
            self.sock.sendto(msg, XR_SOCKET)
        except OSError:
            pass                        # driver went away; poll() notices

    def imu(self, suid, ts, payload):
        data = next((v for n, _, v in self.m.pb_parse(payload) if n == 1), b"")
        if len(data) < 12:
            return
        vec = struct.unpack_from("<3f", data)
        if suid == self.accel:
            self.last_accel = vec
        elif suid == self.gyro:
            self.send(struct.pack("<B7xq3f3f", 1, int(ts * SSC_TICK_NS),
                                  *vec, *self.last_accel))

    def controller(self, raw, left, ts):
        buttons = 0
        for bit, flag in XR_BUTTONS:
            if raw[7] & bit:
                buttons |= flag
        for bit, flag in XR_TOUCHES:
            if raw[8] & bit:
                buttons |= flag
        # Byte 5 = stick right (+), byte 4 = stick up (+).
        stick_x = max(-1.0, min(1.0, (raw[5] - 128) / 127.0))
        stick_y = max(-1.0, min(1.0, (raw[4] - 128) / 127.0))
        # Inputs only on change (plus a 50 ms keep-alive): the packets come at
        # ~500 Hz per controller, the IMU message below goes out for each one.
        state = (buttons, raw[3], raw[6], raw[4], raw[5])
        now = time.monotonic()
        if state != self.last_state.get(left) or now - self.last_state_t.get(left, 0.0) > 0.05:
            self.last_state[left] = state
            self.last_state_t[left] = now
            self.send(struct.pack("<BB2xI4f", 2, 0 if left else 1, buttons,
                                  raw[3] / 255.0, raw[6] / 255.0, stick_x, stick_y))
        gyro, accel, us = ctrl_imu(raw)
        self.send(struct.pack("<BB2xIq3f3f", 6, 0 if left else 1, us, int(ts * SSC_TICK_NS),
                              *gyro, *accel))


def load_ssc():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("ssc", os.path.join(here, "ssc-ble-mcu.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    debug = "-v" in sys.argv
    xr_enabled = "--xr" in sys.argv
    ssc_mod = load_ssc()
    ssc = ssc_mod.Ssc()
    suids = ssc_mod.lookup(ssc, "ble_mcu")
    if not suids:
        sys.exit("ble_mcu sensor not found on the SLPI")
    ssc.send(ssc_mod.client_request(suids[0], ssc_mod.MSGID_STD_SENSOR_CONFIG,
                                    ssc_mod.pb_float(1, 500.0)))
    pads = {True: Pad("PICO 4 Controller Left", True),
            False: Pad("PICO 4 Controller Right", False)}
    ble_mcu = suids[0]
    xr = XrBridge(ssc_mod, ssc) if xr_enabled else None
    print("controllers ready", flush=True)
    dump_check, dumping = 0.0, False
    last = {}
    try:
        while True:
            if xr:
                xr.poll()
            for suid, mid, ts, payload in ssc.events(1.0):
                if mid != ssc_mod.MSGID_SENSOR_EVENT:
                    continue
                if suid != ble_mcu:
                    if xr:
                        xr.imu(suid, ts, payload)
                    continue
                raw = next((v for n, _, v in ssc_mod.pb_parse(payload) if n == 1), b"")
                if len(raw) < 60 or raw[0] != 0xA5 or not packet_valid(raw):
                    continue
                if time.monotonic() >= dump_check:
                    # touch DUMP_TRIGGER: checked once a second, not per packet
                    dump_check = time.monotonic() + 1.0
                    dumping = os.path.exists(DUMP_TRIGGER)
                if dumping:
                    with open(DUMP_FILE, "a") as dump:
                        dump.write("%.3f %d %s\n" % (time.monotonic(), ts, raw.hex()))
                left = bool(raw[7] & 0x40)
                pads[left].update(raw)
                if xr:
                    xr.controller(raw, left, ts)
                if debug and last.get(left) != raw[3:9]:
                    last[left] = raw[3:9]
                    print("%s trig=%3d grip=%3d stick=%3d,%3d b7=%02x b8=%02x" %
                          ("L" if left else "R", raw[3], raw[6], raw[4], raw[5],
                           raw[7], raw[8]), flush=True)
    finally:
        for p in pads.values():
            p.close()


if __name__ == "__main__":
    main()
