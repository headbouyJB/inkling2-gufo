#!/usr/bin/env -S python3 -I
"""Synthetic test images for the M5 image checks (pure stdlib; writes PNGs).
usage: gen_images.py OUTDIR
  red_square.png   320x320  one red square on white
  blue_circle.png  400x240  one blue disc on a yellow background
  ink42.png        480x200  black "INK 42" on white (5x7 glyphs, scale 12)
  big_text.png     1280x720 "INK 42" and three coloured shapes (576 image rows)
  size_2560x1600.png, size_3200x2000.png   large images for timing (2560 and 4000 image rows)
"""
import struct, sys, zlib, os

def png(path, w, h, px):
    raw = b''.join(b'\x00' + bytes(px[y * w * 3:(y + 1) * w * 3]) for y in range(h))
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))

class Img:
    def __init__(s, w, h, bg):
        s.w, s.h, s.px = w, h, bytearray(bytes(bg) * (w * h))
    def rect(s, x0, y0, x1, y1, c):
        x0, x1 = max(x0, 0), min(x1, s.w)
        if x1 <= x0: return
        row = bytes(c) * (x1 - x0)
        for y in range(max(y0, 0), min(y1, s.h)):
            s.px[(y * s.w + x0) * 3:(y * s.w + x1) * 3] = row
    def disc(s, cx, cy, r, c):
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r and 0 <= x < s.w and 0 <= y < s.h:
                    s.px[(y * s.w + x) * 3:(y * s.w + x) * 3 + 3] = bytes(c)
    def text(s, x0, y0, t, scale, c):
        for ch in t:
            g = GLYPH.get(ch)
            if g:
                for row in range(7):
                    for col in range(5):
                        if g[row] & (0x10 >> col):
                            s.rect(x0 + col * scale, y0 + row * scale, x0 + (col + 1) * scale, y0 + (row + 1) * scale, c)
            x0 += 6 * scale
    def save(s, path): png(path, s.w, s.h, s.px)

GLYPH = {
    'I': [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F],
    'N': [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
    'K': [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
    '4': [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    '2': [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
    ' ': [0] * 7,
}

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
im = Img(320, 320, (255, 255, 255)); im.rect(80, 80, 240, 240, (220, 20, 20)); im.save(out + '/red_square.png')
im = Img(400, 240, (250, 220, 40)); im.disc(200, 120, 80, (20, 40, 220)); im.save(out + '/blue_circle.png')
im = Img(480, 200, (255, 255, 255)); im.text(30, 50, 'INK 42', 12, (0, 0, 0)); im.save(out + '/ink42.png')
im = Img(1280, 720, (235, 235, 235)); im.text(60, 60, 'INK 42', 24, (0, 0, 0))
im.rect(100, 380, 400, 640, (220, 20, 20)); im.disc(700, 510, 130, (20, 160, 40)); im.rect(900, 400, 1200, 620, (20, 40, 220))
im.save(out + '/big_text.png')
for name, w, h in (('size_2560x1600', 2560, 1600), ('size_3200x2000', 3200, 2000)):  # 2560 and 4000 image rows
    im = Img(w, h, (235, 235, 235)); im.text(w // 20, h // 20, 'INK 42', w // 53, (0, 0, 0))
    im.rect(w // 8, h // 2, w // 3, h * 9 // 10, (220, 20, 20)); im.rect(w // 2, h // 2, w * 3 // 4, h * 9 // 10, (20, 40, 220))
    im.save(f'{out}/{name}.png')
print('wrote', out)
