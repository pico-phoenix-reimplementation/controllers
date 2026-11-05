#!/usr/bin/env python3
"""
Minimal Qualcomm SSC (SEE) client over QRTR/QMI, no protobuf library needed.

Used to reach PICO's SLPI sensor "ble_mcu" (sns_ble_mcu in the SLPI image),
which carries the controller packets from the Nordic station MCU.

  ssc-ble-mcu.py lookup [data_type ...]     SUID lookup (default: ble_mcu accel)
  ssc-ble-mcu.py attrs <data_type>           print the sensor attributes
  ssc-ble-mcu.py stream <data_type> [secs] [rate] [msgid]
                                             enable and dump raw events

Needs the SLPI running with hexagonrpcd serving the sensor registry (AGENTS §R).
"""
import os
import select
import socket
import struct
import sys
import time

AF_QIPCRTR = 42
QRTR_PORT_CTRL = 0xFFFFFFFE
QRTR_TYPE_NEW_SERVER = 4
QRTR_TYPE_NEW_LOOKUP = 10

SSC_SERVICE = 400
QMI_REQ, QMI_RESP, QMI_IND = 0, 2, 4
SNS_CLIENT_REQ = 0x20
SNS_CLIENT_REPORT_IND = 0x21
SNS_CLIENT_JUMBO_REPORT_IND = 0x22

LOOKUP_SUID = (0xABABABABABABABAB, 0xABABABABABABABAB)
MSGID_SUID_REQ = 512
MSGID_SUID_EVENT = 768
MSGID_ATTR_REQ = 1
MSGID_DISABLE_REQ = 10
MSGID_ATTR_EVENT = 128
MSGID_STD_SENSOR_CONFIG = 513
MSGID_ON_CHANGE_CONFIG = 514
MSGID_ERROR_EVENT = 1022
MSGID_SENSOR_EVENT = 1025

ATTR_NAMES = {0: "name", 1: "vendor", 2: "type", 3: "version", 4: "available",
              5: "api", 6: "rates", 7: "resolutions", 8: "fifo_size",
              9: "active_current", 10: "sleep_current", 11: "ranges",
              12: "op_modes", 13: "dri", 14: "stream_sync", 15: "event_size",
              16: "stream_type", 17: "dynamic", 18: "hw_id", 19: "rigid_body",
              20: "placement", 21: "physical_sensor", 22: "physical_sensor_tests",
              23: "selected_resolution", 24: "selected_range",
              25: "additional_low_latency_rates", 26: "passive_request"}


# ---------------------------------------------------------------- protobuf --
def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def pb_field(num, wt, val):
    key = varint((num << 3) | wt)
    if wt == 0:
        return key + varint(val)
    if wt == 1:
        return key + struct.pack("<Q", val)
    if wt == 5:
        return key + struct.pack("<I", val)
    return key + varint(len(val)) + val      # wt 2


def pb_float(num, f):
    return varint((num << 3) | 5) + struct.pack("<f", f)


def pb_parse(buf):
    """Return a list of (field, wiretype, value)."""
    i, out = 0, []
    while i < len(buf):
        key, n = 0, 0
        while True:
            b = buf[i]
            i += 1
            key |= (b & 0x7F) << n
            n += 7
            if b < 0x80:
                break
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, n = 0, 0
            while True:
                b = buf[i]
                i += 1
                v |= (b & 0x7F) << n
                n += 7
                if b < 0x80:
                    break
        elif wt == 1:
            v = struct.unpack_from("<Q", buf, i)[0]
            i += 8
        elif wt == 5:
            v = buf[i:i + 4]
            i += 4
        elif wt == 2:
            ln, n = 0, 0
            while True:
                b = buf[i]
                i += 1
                ln |= (b & 0x7F) << n
                n += 7
                if b < 0x80:
                    break
            v = buf[i:i + ln]
            i += ln
        else:
            raise ValueError("wire type %d" % wt)
        out.append((num, wt, v))
    return out


def suid_msg(low, high):
    return pb_field(1, 1, low) + pb_field(2, 1, high)


def client_request(suid, msg_id, payload=b"", batch=None, passive=False):
    std = b""
    if batch is not None:
        std += pb_field(1, 2, pb_field(1, 5, batch))
    std += pb_field(2, 2, payload)
    if passive:
        std += pb_field(3, 0, 1)
    susp = pb_field(1, 0, 1) + pb_field(2, 0, 1)   # APSS, no-wakeup
    return (pb_field(1, 2, suid_msg(*suid)) + pb_field(2, 5, msg_id) +
            pb_field(3, 2, susp) + pb_field(4, 2, std))


# ----------------------------------------------------------------- QMI/QRTR --
class Ssc:
    def __init__(self):
        self.sock = socket.socket(AF_QIPCRTR, socket.SOCK_DGRAM)
        self.local_node = self.sock.getsockname()[0]
        self.addr = self._lookup(SSC_SERVICE)
        self.txn = 1

    def _lookup(self, service):
        ctrl = socket.socket(AF_QIPCRTR, socket.SOCK_DGRAM)
        node = ctrl.getsockname()[0]
        ctrl.sendto(struct.pack("<5I", QRTR_TYPE_NEW_LOOKUP, service, 0, 0, 0),
                    (node, QRTR_PORT_CTRL))
        end = time.monotonic() + 3
        while time.monotonic() < end:
            r, _, _ = select.select([ctrl], [], [], 0.5)
            if not r:
                continue
            pkt = ctrl.recv(64)
            cmd, svc, inst, snode, sport = struct.unpack_from("<5I", pkt)
            if cmd == QRTR_TYPE_NEW_SERVER and svc == service:
                ctrl.close()
                print("SSC service %d at %d:%d" % (service, snode, sport))
                return (snode, sport)
        sys.exit("SSC QMI service not found (is the SLPI up?)")

    def send(self, req, jumbo=True):
        tlv = struct.pack("<BH", 1, len(req) + 2) + struct.pack("<H", len(req)) + req
        if jumbo:
            tlv += struct.pack("<BHB", 0x10, 1, 1)
        hdr = struct.pack("<BHHH", QMI_REQ, self.txn, SNS_CLIENT_REQ, len(tlv))
        self.txn += 1
        self.sock.sendto(hdr + tlv, self.addr)

    def recv(self, timeout):
        """Yield (msg_type, msg_id, tlvs) for QMI packets until timeout."""
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            r, _, _ = select.select([self.sock], [], [], left)
            if not r:
                continue
            # Drain everything queued: ~2000 indications/s (controllers + head
            # IMU), one select() per packet cost a large part of a core.
            pkt = self.sock.recv(65536)
            while pkt is not None:
                mtype, txn, mid, ln = struct.unpack_from("<BHHH", pkt)
                tlvs, i = {}, 7
                while i < 7 + ln:
                    t, l = struct.unpack_from("<BH", pkt, i)
                    tlvs[t] = pkt[i + 3:i + 3 + l]
                    i += 3 + l
                yield mtype, mid, tlvs
                try:
                    pkt = self.sock.recv(65536, socket.MSG_DONTWAIT)
                except BlockingIOError:
                    pkt = None

    def events(self, timeout):
        """Yield (suid, msg_id, timestamp, payload) from report indications."""
        for mtype, mid, tlvs in self.recv(timeout):
            if mtype == QMI_RESP:
                res = tlvs.get(2, b"\0\0\0\0")
                if struct.unpack_from("<H", res)[0]:
                    print("QMI error response:", tlvs)
                continue
            if mtype != QMI_IND or mid not in (SNS_CLIENT_REPORT_IND,
                                               SNS_CLIENT_JUMBO_REPORT_IND):
                continue
            raw = tlvs.get(2)
            if raw is None:
                continue
            msg = raw[2:2 + struct.unpack_from("<H", raw)[0]]
            suid = None
            for num, wt, v in pb_parse(msg):
                if num == 1:
                    f = {n: x for n, _, x in pb_parse(v)}
                    suid = (f.get(1), f.get(2))
                elif num == 2:
                    ev = {n: x for n, _, x in pb_parse(v)}
                    yield (suid, struct.unpack("<I", ev[1])[0]
                           if isinstance(ev[1], bytes) else ev[1],
                           ev.get(2), ev.get(3, b""))


def lookup(ssc, data_type, timeout=3):
    req = pb_field(1, 2, data_type.encode()) + pb_field(3, 0, 0)
    ssc.send(client_request(LOOKUP_SUID, MSGID_SUID_REQ, req))
    found = []
    for suid, mid, ts, payload in ssc.events(timeout):
        if mid != MSGID_SUID_EVENT:
            continue
        fields = pb_parse(payload)
        dt = next((v.decode() for n, _, v in fields if n == 1), "")
        if dt != data_type:
            continue
        for n, _, v in fields:
            if n == 2:
                f = {k: x for k, _, x in pb_parse(v)}
                found.append((f[1], f[2]))
        break
    return found


def attrs(ssc, suid, timeout=3):
    ssc.send(client_request(suid, MSGID_ATTR_REQ))
    for s, mid, ts, payload in ssc.events(timeout):
        if mid != MSGID_ATTR_EVENT or s != suid:
            continue
        out = {}
        for n, _, a in pb_parse(payload):
            f = pb_parse(a)
            aid = next(v for k, _, v in f if k == 1)
            vals = []
            for k, _, val in f:
                if k != 2:
                    continue
                for _, _, item in pb_parse(val):
                    for vk, vwt, vv in pb_parse(item):
                        if vk == 2:
                            vals.append(vv.decode(errors="replace"))
                        elif vk == 3:
                            vals.append(round(struct.unpack("<f", vv)[0], 3))
                        elif vk in (4, 5):
                            vals.append(vv)
                        else:
                            vals.append(vv.hex() if isinstance(vv, bytes) else vv)
            out[ATTR_NAMES.get(aid, aid)] = vals
        return out
    return None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "lookup"
    ssc = Ssc()

    if mode == "lookup":
        for dt in sys.argv[2:] or ["ble_mcu", "accel"]:
            suids = lookup(ssc, dt)
            print("%-12s %s" % (dt, ["%016x%016x" % (h, l) for l, h in suids]
                                or "not found"))
        return

    dt = sys.argv[2]
    suids = lookup(ssc, dt)
    if not suids:
        sys.exit("%s: no SUID" % dt)
    suid = suids[0]
    a = attrs(ssc, suid)
    for k, v in (a or {}).items():
        if k != "api":
            print("  %s: %s" % (k, v))
    if mode == "attrs":
        return

    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10
    rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    msgid = int(sys.argv[5]) if len(sys.argv) > 5 else (
        MSGID_STD_SENSOR_CONFIG if rate else MSGID_ON_CHANGE_CONFIG)
    payload = pb_float(1, rate) if msgid == MSGID_STD_SENSOR_CONFIG else b""
    print("enable %s with msg %d rate %s" % (dt, msgid, rate))
    ssc.send(client_request(suid, msgid, payload))
    counts = {}
    shown = 0
    dump = open(os.environ["SSC_DUMP"], "w") if os.environ.get("SSC_DUMP") else None
    for s, mid, ts, p in ssc.events(secs):
        if dump and mid == MSGID_SENSOR_EVENT:
            # payload = field 1 (bytes): the raw MCU packet
            raw = next((v for n, _, v in pb_parse(p) if n == 1), p)
            dump.write("%.3f %d %s\n" % (time.monotonic(), ts, raw.hex()))
        counts[mid] = counts.get(mid, 0) + 1
        if shown < 40 or mid not in (MSGID_SENSOR_EVENT,):
            print("%d ts=%s len=%d %s" % (mid, ts, len(p), p[:64].hex(" ")))
            shown += 1
    print("event counts:", counts)


if __name__ == "__main__":
    main()
