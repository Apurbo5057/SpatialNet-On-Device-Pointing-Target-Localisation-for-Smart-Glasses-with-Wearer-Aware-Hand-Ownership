"""Thumbnails for the architecture figure, drawn from one test image's ground-truth labels.
Run from the repository root: python paper/figures/make_thumbnails.py
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from annotation import parse_annotation  # noqa: E402

NAME = 'img_0000557'
DATA = os.path.join(ROOT, 'EgoPoint_spatialnet', 'test')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thumbs')
HAND, TIP, DIRC, TARGET = (42, 120, 214), (210, 88, 31), (74, 58, 167), (18, 128, 90)
SCALE = 3                                      # draw at 3x, then downsample for smooth lines

ann = parse_annotation(os.path.join(DATA, 'labels', NAME + '.txt'))
base = Image.open(os.path.join(DATA, 'images', NAME + '.jpg')).convert('RGB')
W = base.size[0] * SCALE
big = base.resize((W, W), Image.LANCZOS)
px = lambda v: v * W


def box(d, b, col, width=3, dash=False):
    x1, y1, x2, y2 = px(b[0] - b[2] / 2), px(b[1] - b[3] / 2), px(b[0] + b[2] / 2), px(b[1] + b[3] / 2)
    w = width * SCALE
    if not dash:
        d.rectangle([x1, y1, x2, y2], outline=col, width=w)
        return
    seg = 7 * SCALE
    for (a, b2) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = int(max(abs(b2[0] - a[0]), abs(b2[1] - a[1])) // seg)
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1)
            d.line([a[0] + (b2[0] - a[0]) * t0, a[1] + (b2[1] - a[1]) * t0,
                    a[0] + (b2[0] - a[0]) * t1, a[1] + (b2[1] - a[1]) * t1], fill=col, width=w)


def dot(d, p, col, r=5):
    r *= SCALE
    d.ellipse([px(p[0]) - r, px(p[1]) - r, px(p[0]) + r, px(p[1]) + r], fill=col, outline=(255, 255, 255),
              width=2 * SCALE)


def arrow(d, p, u, length, col, width=4):
    x0, y0 = px(p[0]), px(p[1])
    x1, y1 = x0 + u[0] * length * W, y0 + u[1] * length * W
    d.line([x0, y0, x1, y1], fill=col, width=width * SCALE)
    nx, ny = -u[1], u[0]
    h = 12 * SCALE
    d.polygon([(x1 + u[0] * h * 0.4, y1 + u[1] * h * 0.4), (x1 - u[0] * h + nx * h * 0.55, y1 - u[1] * h + ny * h * 0.55),
               (x1 - u[0] * h - nx * h * 0.55, y1 - u[1] * h - ny * h * 0.55)], fill=col)


def save(img, name):
    os.makedirs(OUT, exist_ok=True)
    img.resize((base.size[0] * 2, base.size[1] * 2), Image.LANCZOS).save(os.path.join(OUT, name + '.png'))


tip, hand, roi = ann['fingertip'], ann['hand'], ann['roi']
u = np.array([roi[0] - tip[0], roi[1] - tip[1]])
u = u / np.linalg.norm(u)

save(big.copy(), 'input')

im = big.copy(); d = ImageDraw.Draw(im)
box(d, hand, HAND); dot(d, hand[:2], HAND, 4)
save(im, 'hand')

im = big.copy(); d = ImageDraw.Draw(im)
box(d, hand, HAND, 2, dash=True); dot(d, tip, TIP, 6)
save(im, 'fingertip')

im = big.copy(); d = ImageDraw.Draw(im)
arrow(d, tip, u, 0.33, DIRC); dot(d, tip, TIP, 6)
save(im, 'direction')

# target: ray-aligned band (illustrates the ray-conditioned prior) + target box
ys, xs = np.mgrid[0:W, 0:W] / W
dx, dy = xs - tip[0], ys - tip[1]
t = dx * u[0] + dy * u[1]
s = np.abs(dx * u[1] - dy * u[0])
band = np.exp(-(s / (0.04 + 0.12 * np.clip(t, 0, None))) ** 2) * (t > 0)
over = np.array(big, dtype=np.float32)
tint = np.array(TARGET, dtype=np.float32)
a = (0.55 * band)[..., None]
im = Image.fromarray((over * (1 - a) + tint * a).astype(np.uint8))
d = ImageDraw.Draw(im)
box(d, roi, TARGET, 3); dot(d, tip, TIP, 5)
save(im, 'target')
print('thumbnails written to', OUT)
