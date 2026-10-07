"""Real-inference assets for the TikZ SpatialNet architecture figure (spatialnet_architecture.tex).

Runs the paper checkpoint on ONE test image on the CPU and writes, into ./assets:
  input.png / output.png          the frame and the same frame with the predictions drawn on it
  hm_hand / hm_tip / hm_target    sigmoid of the 20x20 heatmap logits H^h, H^p, H^t
  off_dx / off_dy                 fingertip-head offset channels alpha_p (g - c_h)
  ray_*.png                       the six target-head ray channels R(p, theta)
  coord_x / coord_y               the two CoordConv channels
  pred.tex                        predicted cells / boxes / angle / gate probabilities as TeX macros
Usage (repo root):  python docs/figures/architecture_tikz/make_arch_assets.py [--name img_0010965] [--preview a,b,c]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use('Agg')
from matplotlib import colormaps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
import config as cfg  # noqa: E402
import model as M  # noqa: E402
from annotation import parse_annotation  # noqa: E402

CKPT = os.path.join(ROOT, 'runs', 'main_run', 'outputs_spatialnet', 'best_model.pth')
TEST = os.path.join(ROOT, 'EgoPoint_spatialnet', 'test')
OUT = os.path.join(HERE, 'assets')
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# figure palette (matches the TikZ colours)
HAND, TIP, DIRC, TARGET = (232, 131, 58), (46, 160, 103), (123, 94, 167), (42, 99, 184)


def load_model():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    m = M.SpatialNet(load_pretrained=False)
    m.load_state_dict(ck.get('model_state_dict', ck))
    return m.eval()


def to_tensor(img):
    # the dataset's PIL fallback of the evaluation transform (albumentations is not installed locally)
    x = np.array(Image.fromarray(img).resize((cfg.IMG_SIZE, cfg.IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy((x - MEAN) / STD).permute(2, 0, 1).unsqueeze(0).float()


def run(m, img):
    """Forward pass that also keeps the intermediate maps the figure shows."""
    feats = {}
    hooks = [m.roi_hm.cond.register_forward_hook(lambda mod, i, o: feats.__setitem__('roi_cond_in', i[0].detach())),
             m.ft_hm.cond.register_forward_hook(lambda mod, i, o: feats.__setitem__('ft_cond_in', i[0].detach()))]
    with torch.no_grad():
        o = m(to_tensor(img))
    for h in hooks:
        h.remove()
    o.update(feats)
    return o


def colorize(a, cmap, lo, hi, px=400):
    """20x20 map -> RGB image, nearest-neighbour so every feature cell stays visible."""
    a = np.clip((np.asarray(a, dtype=np.float32) - lo) / max(hi - lo, 1e-8), 0, 1)
    rgb = (colormaps[cmap](a)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((px, px), Image.NEAREST)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def summary(name, o):
    ann = parse_annotation(os.path.join(TEST, 'labels', name + '.txt'))
    hand, tip, roi = o['hand'][0].numpy(), o['fingertip'][0].numpy(), o['roi'][0].numpy()
    sin, cos = o['angle'][0].tolist()
    g = np.array(ann['roi'][:2]) - np.array(ann['fingertip'])
    ang_err = np.degrees(np.arccos(np.clip((cos * g[0] + sin * g[1]) / np.linalg.norm(g), -1, 1)))
    return dict(hand_iou=iou(hand, ann['hand']), target_iou=iou(roi, ann['roi']),
                tip_err_px=float(np.linalg.norm(tip - np.array(ann['fingertip'])) * cfg.IMG_SIZE), dir_err_deg=ang_err,
                p_hand=torch.sigmoid(o['hand_conf']).item(), p_point=torch.sigmoid(o['point_conf']).item(),
                p_target=torch.sigmoid(o['roi_conf']).item())


def draw_output(img, o, size=480):
    S3 = 3 * size
    im = Image.fromarray(img).resize((S3, S3), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    px = lambda v: float(v) * S3

    def box(b, col):
        d.rectangle([px(b[0] - b[2] / 2), px(b[1] - b[3] / 2), px(b[0] + b[2] / 2), px(b[1] + b[3] / 2)],
                    outline=col, width=20)

    hand, tip, roi = o['hand'][0].numpy(), o['fingertip'][0].numpy(), o['roi'][0].numpy()
    sin, cos = o['angle'][0].tolist()
    box(hand, HAND)
    box(roi, TARGET)
    # direction arrow from the fingertip along the predicted unit vector
    L = max(0.12, 0.8 * float(np.hypot(roi[0] - tip[0], roi[1] - tip[1])))
    x0, y0 = px(tip[0]), px(tip[1])
    x1, y1 = x0 + cos * L * S3, y0 + sin * L * S3
    h = 60
    d.line([x0, y0, x1 - cos * h * 0.6, y1 - sin * h * 0.6], fill=DIRC, width=22)
    nx, ny = -sin, cos
    d.polygon([(x1, y1), (x1 - cos * h + nx * h * 0.6, y1 - sin * h + ny * h * 0.6),
               (x1 - cos * h - nx * h * 0.6, y1 - sin * h - ny * h * 0.6)], fill=DIRC)
    r = 24
    d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=TIP, outline=(255, 255, 255), width=6)
    return im.resize((size, size), Image.LANCZOS)


def export(name, m):
    os.makedirs(OUT, exist_ok=True)
    img = np.array(Image.open(os.path.join(TEST, 'images', name + '.jpg')).convert('RGB'))
    o = run(m, img)
    G = o['hand_hm'].shape[-1]
    Image.fromarray(img).resize((480, 480), Image.LANCZOS).save(os.path.join(OUT, 'input.png'))
    draw_output(img, o).save(os.path.join(OUT, 'output.png'))

    for key, fn in (('hand_hm', 'hm_hand'), ('ft_hm', 'hm_tip'), ('roi_hm', 'hm_target')):
        colorize(torch.sigmoid(o[key])[0, 0].numpy(), 'turbo', 0.0, 1.0).save(os.path.join(OUT, fn + '.png'))

    # fingertip offset channels (dx, dy): the last two channels entering G^p
    ft_in = o['ft_cond_in'][0]
    for c, fn in ((-2, 'off_dx'), (-1, 'off_dy')):
        a = ft_in[c].numpy()
        v = np.abs(a).max()
        colorize(a, 'coolwarm', -v, v).save(os.path.join(OUT, fn + '.png'))
    # six ray channels dx, dy, t, |s|, cos, sin: the last six channels entering G^t
    ray = o['roi_cond_in'][0, -6:].numpy()
    v = max(np.abs(ray[:4]).max(), 1e-6)
    for i, fn in enumerate(('ray_dx', 'ray_dy', 'ray_t', 'ray_s', 'ray_cos', 'ray_sin')):
        colorize(ray[i], 'coolwarm', -(v if i < 4 else 1.0), v if i < 4 else 1.0).save(os.path.join(OUT, fn + '.png'))
    lin = np.linspace(-1, 1, G, dtype=np.float32)
    colorize(np.tile(lin, (G, 1)), 'coolwarm', -1, 1).save(os.path.join(OUT, 'coord_x.png'))
    colorize(np.tile(lin[:, None], (1, G)), 'coolwarm', -1, 1).save(os.path.join(OUT, 'coord_y.png'))

    hand, tip, roi = o['hand'][0].numpy(), o['fingertip'][0].numpy(), o['roi'][0].numpy()
    sin, cos = o['angle'][0].tolist()
    s = summary(name, o)
    # model convention: normalised coordinate x <-> feature cell index x * (G - 1)
    cellf = lambda q: float(q) * (G - 1)
    peak = lambda k: divmod(int(o[k][0, 0].flatten().argmax()), G)   # (row, col) of the arg-max cell k*
    hr, hc = peak('hand_hm')
    pr, pc = peak('ft_hm')
    tr, tc = peak('roi_hm')
    macros = {
        'ImgName': name.replace('_', r'\_'),
        'HandCx': cellf(hand[0]), 'HandCy': cellf(hand[1]), 'HandW': hand[2] * (G - 1), 'HandH': hand[3] * (G - 1),
        'TipX': cellf(tip[0]), 'TipY': cellf(tip[1]),
        'TgtCx': cellf(roi[0]), 'TgtCy': cellf(roi[1]), 'TgtW': roi[2] * (G - 1), 'TgtH': roi[3] * (G - 1),
        'HandPkR': hr, 'HandPkC': hc, 'TipPkR': pr, 'TipPkC': pc, 'TgtPkR': tr, 'TgtPkC': tc,
        'ThetaDeg': float(np.degrees(np.arctan2(sin, cos))),
        'PHand': s['p_hand'], 'PPoint': s['p_point'], 'PTarget': s['p_target'],
        'HandIoU': s['hand_iou'], 'TgtIoU': s['target_iou'], 'TipErr': s['tip_err_px'], 'DirErr': s['dir_err_deg'],
        'NParams': m.count_parameters() / 1e6,
    }
    with open(os.path.join(OUT, 'pred.tex'), 'w') as f:
        f.write('% generated by make_arch_assets.py; feature-cell units on the 20x20 grid, (0,0) = top-left cell\n')
        for k, val in macros.items():
            f.write('\\def\\%s{%s}\n' % (k, val if isinstance(val, (str, int)) else '%.3f' % val))
    print(name, {k: round(val, 3) for k, val in s.items()}, 'params=%d' % m.count_parameters())


def preview(names, m, path):
    tiles = []
    for n in names:
        img = np.array(Image.open(os.path.join(TEST, 'images', n + '.jpg')).convert('RGB'))
        o = run(m, img)
        row = [draw_output(img, o, 200)] + [colorize(torch.sigmoid(o[k])[0, 0].numpy(), 'turbo', 0, 1, px=200)
                                            for k in ('hand_hm', 'ft_hm', 'roi_hm')]
        tiles.append(row)
        print(n, {k: round(val, 3) for k, val in summary(n, o).items()})
    sheet = Image.new('RGB', (4 * 205, len(tiles) * 205), 'white')
    for r, row in enumerate(tiles):
        for c, t in enumerate(row):
            sheet.paste(t, (c * 205, r * 205))
    sheet.save(path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='img_0010965')
    ap.add_argument('--preview', default='', help='comma-separated image names; writes a heatmap contact sheet only')
    ap.add_argument('--preview_out', default='arch_candidates.png')
    a = ap.parse_args()
    torch.set_num_threads(2)
    net = load_model()
    if a.preview:
        preview(a.preview.split(','), net, a.preview_out)
    else:
        export(a.name, net)
