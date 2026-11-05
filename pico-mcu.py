#!/usr/bin/env python3
"""
PICO 4 station MCU (Nordic, SPI19) helper for mainline Linux.

Protocol (captured from stock /system/bin/stationservice, see AGENTS.md):
  * fixed 40-byte frames, plain write()/read() on spidev
  * host -> MCU: aa 01 <cmd> <payload...>, byte 37 = index (0 for commands),
    byte 39 = CRC-8/MAXIM over bytes 0..38
  * MCU -> host: bb .. (reply) / a5 .. (event, a5 07 = ASCII log text)
  * the MCU raises TLMM 11 when it has a frame; every received frame is
    acked with aa 01 fa <rx[37]>

Camera session (stock): 19 02 00 01, then 04 01 00 02 (init, MCU programs
the four OV7251s), 04 02 00 01 (stream on), then 04 03 exposure updates at
the frame rate. The OV7251s must already be powered (pico-ov7251-mcu).

Controller calibration (stock GetCV3Calibration): aa <ch> 02 <block 1..12> on
channel 2 (left) / 3 (right); the reply bb <ch> 02 <block> carries 8 float32
(bb <ch> 02 ff = not available yet, ask again). The 96 floats are two IMU
calibrations (3x3 + 3 each, floats 0..23), the 20 IR LED positions in metres
in the controller IMU frame (24..83) and an LED -> IMU pose (84..95: xyz + pad,
quaternion xyzw). "start" saves them to CALIB_DIR/controller_{left,right}.json.

Controller commands seen in stock (channel 2/3): 04 01 01 00 01 0f once per
session (key/IMU stream on), 07 01 21 05 00 b1 on UI events (haptic pulse).

Usage: pico-mcu.py monitor | start [seconds] | calib
"""
import fcntl
import glob
import json
import os
import select
import struct
import sys
import threading
import time

FRAME = 40
CALIB_DIR = os.environ.get("PICO4_CTRL_CALIB_DIR", "/run/pico4")
SPEED_HZ = 6000000

SPI_IOC_WR_MODE = 0x40016B01
SPI_IOC_WR_BITS_PER_WORD = 0x40016B03
SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04

GPIO_GET_CHIPINFO_IOCTL = 0x8044B401
GPIO_V2_GET_LINE_IOCTL = 0xC250B407
GPIO_V2_LINE_FLAG_INPUT = 1 << 2
GPIO_V2_LINE_FLAG_EDGE_RISING = 1 << 4
GPIO_V2_LINE_EVENT_SIZE = 48

TLMM_LABEL = "f100000.pinctrl"
MCU_VDD_STATE = "/sys/devices/platform/mcu-vdd/state"
READY_LINE = 11

# One stock 04 03 frame: four (u16 exposure, u8 gain) groups, a0 00, counter.
EXPOSURE = bytes.fromhex("ed0030e40030ea0030ea0030a000")
# PICO_MCU_EXPOSURE="lines,gain" overrides all four cameras (e.g. "800,128").
if os.environ.get("PICO_MCU_EXPOSURE"):
    _e, _g = (int(x, 0) for x in os.environ["PICO_MCU_EXPOSURE"].split(","))
    EXPOSURE = struct.pack("<HB", _e, _g) * 4 + b"\xa0\x00"


def crc8_maxim(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if crc & 1 else crc >> 1
    return crc


def frame(cmd, payload=b"", index=0, chan=1):
    f = bytearray(FRAME)
    f[0] = 0xAA
    f[1] = chan
    f[2] = cmd
    f[3:3 + len(payload)] = payload
    f[37] = index
    f[39] = crc8_maxim(f[:39])
    return bytes(f)


def find_spidev():
    for dev in glob.glob("/sys/bus/spi/devices/spi*"):
        try:
            compat = open(dev + "/of_node/compatible", "rb").read()
        except OSError:
            continue
        if b"picovr,nordic" in compat:
            node = glob.glob(dev + "/spidev/spidev*")
            if node:
                return "/dev/" + os.path.basename(node[0])
    sys.exit("no picovr,nordic spidev found")


def find_tlmm_chip():
    for path in sorted(glob.glob("/dev/gpiochip*")):
        info = bytearray(68)  # struct gpiochip_info: name[32], label[32], lines
        try:
            with open(path, "rb") as chip:
                fcntl.ioctl(chip, GPIO_GET_CHIPINFO_IOCTL, info)
        except OSError:
            continue
        if info[32:64].split(b"\0")[0].decode() == TLMM_LABEL:
            return path
    sys.exit("TLMM gpiochip not found")


def request_ready_line(chip_path):
    req = bytearray(592)
    struct.pack_into("<I", req, 0, READY_LINE)
    req[256:256 + 9] = b"pico-mcu\0"
    struct.pack_into("<Q", req, 288, GPIO_V2_LINE_FLAG_INPUT |
                     GPIO_V2_LINE_FLAG_EDGE_RISING)
    struct.pack_into("<I", req, 560, 1)         # num_lines
    struct.pack_into("<I", req, 564, 64)        # event_buffer_size
    with open(chip_path, "rb") as chip:
        fcntl.ioctl(chip, GPIO_V2_GET_LINE_IOCTL, req)
    return struct.unpack_from("<i", req, 588)[0]


class Mcu:
    def __init__(self):
        self.spi = os.open(find_spidev(), os.O_RDWR)
        fcntl.ioctl(self.spi, SPI_IOC_WR_MODE, struct.pack("B", 0))
        fcntl.ioctl(self.spi, SPI_IOC_WR_BITS_PER_WORD, struct.pack("B", 8))
        fcntl.ioctl(self.spi, SPI_IOC_WR_MAX_SPEED_HZ, struct.pack("<I", SPEED_HZ))
        self.line = request_ready_line(find_tlmm_chip())
        self.lock = threading.Lock()
        self.replies = {}
        self.text = b""

    def send(self, f):
        with self.lock:
            os.write(self.spi, f)

    def handle(self, rx):
        if rx[0] not in (0xA5, 0xBB):
            return
        if crc8_maxim(rx[:39]) != rx[39]:
            print("crc error:", rx.hex(" "))
            return
        self.send(frame(0xFA, bytes([rx[37]])))
        if rx[:2] == b"\xa5\x07":
            self.text += rx[3:36].split(b"\0")[0]
            while b"\n" in self.text:
                line, self.text = self.text.split(b"\n", 1)
                print("MCU:", line.decode("latin1").strip())
            return
        if rx[0] == 0xBB:
            self.replies[(rx[1], rx[2])] = rx
        print("rx:", rx.hex(" "))

    def poll(self, timeout):
        """Service MCU frames until timeout (seconds)."""
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            r, _, _ = select.select([self.line], [], [], left)
            if not r:
                continue
            os.read(self.line, GPIO_V2_LINE_EVENT_SIZE * 16)
            with self.lock:
                rx = os.read(self.spi, FRAME)
            self.handle(rx)

    def command(self, cmd, payload=b"", timeout=1.0, chan=1):
        key = (chan, cmd)
        self.replies.pop(key, None)
        f = frame(cmd, payload, chan=chan)
        print("tx:", f.hex(" "))
        self.send(f)
        end = time.monotonic() + timeout
        while key not in self.replies and time.monotonic() < end:
            self.poll(0.05)
        reply = self.replies.get(key)
        if reply is None:
            print("no reply to cmd 0x%02x on channel %d" % (cmd, chan))
        return reply

    def controller_calib(self, chan):
        """The 96 calibration floats of one controller, or None."""
        floats = []
        for block in range(1, 13):
            # ~4 s for a controller to answer at all, then a few retries per block.
            for _ in range(12 if block == 1 else 5):
                r = self.command(0x02, bytes([block]), 0.3, chan)
                if r is not None and r[3] == block:
                    floats += struct.unpack_from("<8f", r, 4)
                    break
                self.poll(0.05)         # bb .. 02 ff: not there yet (controller asleep?)
            else:
                print("controller on channel %d: no calibration block %d" % (chan, block))
                return None
        return floats


def save_controller_calib(mcu):
    os.makedirs(CALIB_DIR, exist_ok=True)
    for chan, name in ((2, "left"), (3, "right")):
        # Stock order: 34 <n> before reading controller n, 2b <n> 01 after it; without
        # 2b the controller stays in this mode and stops streaming its keys and IMU
        # to the SLPI (its LEDs keep flashing).
        mcu.command(0x34, bytes([chan - 1]))
        f = mcu.controller_calib(chan)
        mcu.command(0x2B, bytes([chan - 1, 1]))
        if f is None:
            continue
        data = {"imu_a": f[0:12], "imu_b": f[12:24],
                "leds": [f[i:i + 3] for i in range(24, 84, 3)],
                "led_to_imu_position": f[84:87], "led_to_imu_orientation": f[88:92],
                "raw": f}
        path = os.path.join(CALIB_DIR, "controller_%s.json" % name)
        with open(path + ".tmp", "w") as out:
            json.dump(data, out)
        os.replace(path + ".tmp", path)
        print("saved", path)


def mcu_power_cycle():
    """Stock stationservice powers the MCU off and on before a session."""
    if not os.path.exists(MCU_VDD_STATE):
        print("no", MCU_VDD_STATE, "- skipping MCU power cycle")
        return
    for state, wait in (("0", 0.1), ("1", 1.0)):
        with open(MCU_VDD_STATE, "w") as f:
            f.write(state)
        time.sleep(wait)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "monitor"
    if mode == "start" and not os.environ.get("PICO_MCU_NO_CYCLE"):
        mcu_power_cycle()
    mcu = Mcu()

    if mode == "cmds":
        # pico-mcu.py cmds "34 02" "2b 02 01" ...: send raw commands, then monitor
        for c in sys.argv[2:]:
            b = bytes.fromhex(c)
            mcu.command(b[0], b[1:])
        mcu.poll(5)
        return

    if mode == "monitor":
        mcu.poll(float(sys.argv[2]) if len(sys.argv) > 2 else 1e9)
        return

    if mode == "calib":
        mcu.poll(0.5)
        save_controller_calib(mcu)
        return

    if mode != "start":
        sys.exit(__doc__)

    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    mcu.poll(0.5)
    # Stock order around a tracking session (spispy capture at 848-850 s).
    # Stock first runs the idle work mode (19 01) after power-up, then the
    # tracking one. Without the idle phase the controllers connect and flash
    # their LEDs but send no keys/IMU to the SLPI (found 2026-10-25); once
    # they have seen it they keep streaming across station resets.
    mcu.command(0x19, b"\x01")                  # idle work mode
    mcu.command(0x01)
    mcu.poll(2.0)                               # controllers reconnect (a5 02 0N)
    mcu.command(0x19, b"\x02\x00\x01")          # work mode used for tracking
    mcu.command(0x01)                           # status
    # Controller calibration first, in stock's order (34 <n>, blocks, 2b <n> 01).
    if not os.environ.get("PICO_MCU_NO_CALIB"):
        save_controller_calib(mcu)
    mcu.command(0x29, b"\x01")
    mcu.command(0x30)
    # Per controller (channel 2 left, 3 right): switch on its key/IMU stream to
    # the SLPI (stock spispy-3.log, 104.01 s; no reply). A controller keeps the
    # setting until it resets, so sessions only worked while Android had set it.
    for chan in (2, 3):
        f = frame(0x04, b"\x01\x01\x00\x01\x0f", chan=chan)
        print("tx:", f.hex(" "), flush=True)
        mcu.send(f)
        mcu.poll(0.05)
    mcu.command(0x22, b"\x01")
    mcu.command(0x26, b"\x02\x00\xb4")
    mcu.command(0x04, b"\x01\x00\x02", 2.0)     # camera init
    mcu.command(0x04, b"\x02\x00\x01")          # stream on
    mcu.command(0x26, b"\x02\x01\xb4")

    end = time.monotonic() + seconds
    ctr = 0
    while time.monotonic() < end:
        payload = b"\x03" + EXPOSURE + bytes([ctr & 0xFF])
        mcu.send(frame(0x04, payload))
        ctr += 2
        mcu.poll(1 / 30)
        inject(mcu)


INJECT = "/run/pico4/mcu-tx"


def inject(mcu):
    """Send frames written to /run/pico4/mcu-tx ("<chan> <cmd> <payload hex>" per line)."""
    try:
        lines = open(INJECT).read().splitlines()
        os.unlink(INJECT)
    except OSError:
        return
    for line in lines:
        w = line.split()
        if len(w) >= 2:
            f = frame(int(w[1], 16), bytes.fromhex("".join(w[2:])), chan=int(w[0], 16))
            print("tx (injected):", f.hex(" "), flush=True)
            mcu.send(f)


if __name__ == "__main__":
    main()
