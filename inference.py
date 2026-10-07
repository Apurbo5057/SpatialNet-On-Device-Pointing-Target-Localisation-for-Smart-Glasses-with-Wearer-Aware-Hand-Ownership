"""Run SpatialNet on a folder of images or a video/camera stream and write annotated results.

    python inference.py --model_path outputs/best_model.pth --image_dir ./examples/images [--heatmaps]
    python inference.py --model_path outputs/best_model.pth --video in.mp4 --out_dir ./examples/video_results
Video mode adds gate hysteresis and EMA smoothing across frames.
"""

import os
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import config as cfg
from train import add_model_args
from test_evaluate import derive_pred_category, load_model

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CAT_NAMES = {0: 'No Hand', 1: 'Non-Pointing', 2: 'Pointing', 3: 'Pointing (empty)'}
CAT_COLORS = {0: 'gray', 1: 'blue', 2: 'green', 3: 'purple'}


def preprocess(img, img_size):
    resized = img.resize((img_size, img_size), Image.BILINEAR)
    arr = (np.array(resized, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0), np.array(resized)


class TemporalSmoother:
    """Hysteresis on the cascade gates plus EMA on the geometric outputs."""

    def __init__(self, hc=0.5, pc=0.5, rc=0.5, delta=0.15, ema=0.6):
        self.tau = {'hc': hc, 'pc': pc, 'rc': rc}
        self.delta = delta
        self.ema = ema
        self.state = {'hc': False, 'pc': False, 'rc': False}
        self.geo = None
        self.category = 0

    def _gate(self, key, prob):
        t = self.tau[key] - (self.delta if self.state[key] else 0.0)
        self.state[key] = bool(prob >= t)
        return self.state[key]

    def update(self, hc, pc, rc, geo):
        h = self._gate('hc', hc)
        p = h and self._gate('pc', pc)
        r = p and self._gate('rc', rc)
        if not h:
            self.state['pc'] = self.state['rc'] = False
        elif not p:
            self.state['rc'] = False
        category = 0 if not h else (1 if not p else (2 if r else 3))
        if category != self.category or self.geo is None:
            self.geo = {k: np.array(v, dtype=np.float32) for k, v in geo.items()}
        else:
            for k, v in geo.items():
                self.geo[k] = self.ema * self.geo[k] + (1 - self.ema) * v
            self.geo['angle'] = self.geo['angle'] / (np.linalg.norm(self.geo['angle']) + 1e-8)
        self.category = category
        return category, self.geo


@torch.no_grad()
def predict(model, tensor, device):
    out = model(tensor.to(device))
    hc = float(torch.sigmoid(out['hand_conf']).item())
    pc = float(torch.sigmoid(out['point_conf']).item())
    rc = float(torch.sigmoid(out['roi_conf']).item())
    geo = {'hand': out['hand'][0].cpu().numpy(),
           'fingertip': out['fingertip'][0].cpu().numpy(),
           'angle': out['angle'][0].cpu().numpy(),
           'roi': out['roi'][0].cpu().numpy()}
    maps = {k: torch.sigmoid(out[k][0, 0]).cpu().numpy() for k in ['hand_hm', 'ft_hm', 'roi_hm']}
    return hc, pc, rc, geo, maps


def draw(ax, img, category, geo, hc, pc, rc, name, img_size, heat=None):
    ax.imshow(img)
    if heat is not None:
        up = np.array(Image.fromarray((heat * 255).astype(np.uint8)).resize((img_size, img_size),
                                                                           Image.BILINEAR))
        ax.imshow(up, cmap='jet', alpha=0.35)
    if category >= 1:
        cx, cy, w, h = geo['hand'] * img_size
        ax.add_patch(patches.Rectangle((cx - w / 2, cy - h / 2), w, h, lw=2,
                                       ec='cyan', fc='none', label='Wearer hand'))
    if category in (2, 3):
        ft = geo['fingertip'] * img_size
        ax.plot(ft[0], ft[1], 'o', color='lime', ms=8, label='Fingertip')
        a = geo['angle']
        ax.arrow(ft[0], ft[1], a[1] * 30, a[0] * 30, head_width=5, fc='yellow', ec='orange', lw=2)
    if category == 2:
        cx, cy, w, h = geo['roi'] * img_size
        ax.add_patch(patches.Rectangle((cx - w / 2, cy - h / 2), w, h, lw=2,
                                       ec='red', fc='none', label='Target'))
    ax.set_title(f'{name} | {CAT_NAMES[category]} (hc={hc:.2f}, pc={pc:.2f}, rc={rc:.2f})',
                 fontsize=10, fontweight='bold', color=CAT_COLORS[category])
    ax.axis('off')
    if category >= 1:
        ax.legend(loc='upper right', fontsize=8)


def run_images(model, args, device):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    paths = sorted(p for e in exts for p in glob.glob(os.path.join(args.image_dir, e))
                   if not p.endswith('_result.png') and '_summary' not in os.path.basename(p))
    if not paths:
        print(f'No images found in {args.image_dir}')
        return
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        tensor, shown = preprocess(img, args.img_size)
        hc, pc, rc, geo, maps = predict(model, tensor, device)
        cat = derive_pred_category(hc, pc, rc, args.hc_thresh, args.pc_thresh, args.rc_thresh)
        name = os.path.basename(p)
        fig, ax = plt.subplots(figsize=(7, 7))
        draw(ax, shown, cat, geo, hc, pc, rc, name, args.img_size,
             maps['hand_hm'] if args.heatmaps else None)
        out = os.path.join(args.out_dir, os.path.splitext(name)[0] + '_result.png')
        plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close(fig)
        results.append((name, shown, cat, geo, hc, pc, rc, maps))
        print(f'{name:<24} {CAT_NAMES[cat]:<18} hc={hc:.2f} pc={pc:.2f} rc={rc:.2f}  '
              f'hand={np.round(geo["hand"], 3)}  ft={np.round(geo["fingertip"], 3)}')

    cols = min(3, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    axes = axes.flatten()
    for ax, (name, shown, cat, geo, hc, pc, rc, maps) in zip(axes, results):
        draw(ax, shown, cat, geo, hc, pc, rc, name, args.img_size,
             maps['hand_hm'] if args.heatmaps else None)
    for ax in axes[len(results):]:
        ax.axis('off')
    plt.suptitle('SpatialNet Inference', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, '_summary_grid.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved results to {args.out_dir}')


def run_video(model, args, device):
    import cv2
    cap = cv2.VideoCapture(0 if args.video == 'camera' else args.video)
    if not cap.isOpened():
        print(f'Cannot open {args.video}')
        return
    os.makedirs(args.out_dir, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(os.path.join(args.out_dir, 'result.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    smoother = TemporalSmoother(args.hc_thresh, args.pc_thresh, args.rc_thresh)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor, _ = preprocess(img, args.img_size)
        hc, pc, rc, geo, _ = predict(model, tensor, device)
        cat, g = smoother.update(hc, pc, rc, geo)
        if cat >= 1:
            cx, cy, w, h = g['hand'] * np.array([W, H, W, H])
            cv2.rectangle(frame, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), (255, 255, 0), 2)
        if cat in (2, 3):
            ft = (g['fingertip'] * np.array([W, H])).astype(int)
            cv2.circle(frame, tuple(ft), 6, (0, 255, 0), -1)
            tip = ft + (np.array([g['angle'][1], g['angle'][0]]) * 0.15 * W).astype(int)
            cv2.arrowedLine(frame, tuple(ft), tuple(tip), (0, 200, 255), 2)
        if cat == 2:
            cx, cy, w, h = g['roi'] * np.array([W, H, W, H])
            cv2.rectangle(frame, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), (0, 0, 255), 2)
        cv2.putText(frame, f'{CAT_NAMES[cat]} {hc:.2f}/{pc:.2f}/{rc:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        n += 1
    cap.release(); writer.release()
    print(f'Wrote {n} frames to {os.path.join(args.out_dir, "result.mp4")}')


def parse_args():
    p = argparse.ArgumentParser(description='SpatialNet inference')
    p.add_argument('--model_path', type=str, default=os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
    p.add_argument('--image_dir', type=str, default='./examples/images')
    p.add_argument('--video', type=str, default=None, help="video file path, or 'camera'")
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--hc_thresh', type=float, default=0.5)
    p.add_argument('--pc_thresh', type=float, default=0.5)
    p.add_argument('--rc_thresh', type=float, default=0.5)
    p.add_argument('--heatmaps', action='store_true', help='overlay the wearer-hand heatmap')
    add_model_args(p)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.out_dir = args.out_dir or ('./examples/results' if args.video is None else './examples/video_results')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, args, device)
    if args.video:
        run_video(model, args, device)
    else:
        run_images(model, args, device)
