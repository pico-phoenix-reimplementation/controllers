#!/usr/bin/env python3
"""
OV7251 tracking-camera RAW10 (Y10P, 640x481) captures -> PNG, no numpy.

  trk2png.py out.png trk0.raw [trk1.raw ...]   last frame of each, side by side

Line 481 is the sensor's embedded-data line and is dropped. Pixels keep their
top 8 bits; the montage is auto-stretched to the 0.5%..99.5% range.
"""
import struct
import sys
import zlib

W, H = 640, 481
STRIDE = W * 10 // 8
FRAME = STRIDE * H


def last_frame(path):
    data = open(path, "rb").read()
    n = len(data) // FRAME
    if not n:
        sys.exit("%s: no complete frame" % path)
    f = data[(n - 1) * FRAME:n * FRAME]
    return [bytes(b for i, b in enumerate(f[y * STRIDE:(y + 1) * STRIDE]) if i % 5 != 4)
            for y in range(H - 1)]


def write_png(path, rows):
    w = len(rows[0])
    raw = b"".join(b"\0" + r for r in rows)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" +
                chunk(b"IHDR", struct.pack(">IIBBBBB", w, len(rows), 8, 0, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main():
    frames = [last_frame(p) for p in sys.argv[2:]]
    rows = [b"".join(f[y] + b"\x80" * 8 for f in frames) for y in range(H - 1)]
    hist = [0] * 256
    for r in rows[::4]:
        for p in r[::4]:
            hist[p] += 1
    total, acc, lo, hi = sum(hist), 0, 0, 255
    for v, c in enumerate(hist):
        acc += c
        if acc < total * 0.005:
            lo = v
        if acc <= total * 0.995:
            hi = v
    span = max(1, hi - lo)
    lut = bytes(min(255, max(0, (v - lo) * 255 // span)) for v in range(256))
    write_png(sys.argv[1], [r.translate(lut) for r in rows])
    print("%s: %d cameras, stretch %d..%d" % (sys.argv[1], len(frames), lo, hi))


if __name__ == "__main__":
    main()
