"""What shape did the learned ray prior G^t take? (main checkpoint, CPU, a few hundred test images)

Per image, Spearman correlation between G^t over the 20x20 grid and
  (a) distance from the predicted fingertip,
  (b) distance from the predicted ray's line (|s|),
  (c) signed distance along the ray (t; positive = in front of the fingertip).
A "proximity" prior correlates negatively with (a); a "ray" prior negatively with (b) and positively with (c).
Run from the repository root: python paper/scripts/ray_prior_shape.py [--n 400]
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import model as M  # noqa: E402
from dataset import get_eval_loader  # noqa: E402
from annotation import fingertip_roi_gap, gap_bin  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.join(ROOT, 'runs', 'main_run', 'outputs_spatialnet', 'best_model.pth'))
ap.add_argument('--n', type=int, default=400)
args = ap.parse_args()
torch.set_num_threads(4)

ck = torch.load(args.model, map_location='cpu', weights_only=False)
m = M.SpatialNet(load_pretrained=False)
m.load_state_dict(ck.get('model_state_dict', ck))
m.eval()
G = m.grid
ys, xs = np.mgrid[0:G, 0:G] / (G - 1)

dl = get_eval_loader(os.path.join(ROOT, 'EgoPoint_spatialnet'), 'test', batch_size=25, num_workers=0)
rows = []
with torch.no_grad():
    for b in dl:
        keep = (b['roi_conf'].squeeze(1) > 0).nonzero().flatten()
        if len(keep) == 0:
            continue
        out = m(b['image'][keep])
        for j, k in enumerate(keep.tolist()):
            g = out['roi_geo'][j, 0].numpy()
            p = out['fingertip'][j].numpy()
            sin, cos = out['angle'][j].numpy()
            dx, dy = xs - p[0], ys - p[1]
            dist = np.hypot(dx, dy)
            t = dx * cos + dy * sin
            s = np.abs(dx * sin - dy * cos)
            gap = gap_bin(fingertip_roi_gap(b['fingertip_kp'][k].tolist(), b['roi_bbox'][k].tolist()))
            rows.append((gap, spearmanr(g.ravel(), dist.ravel())[0], spearmanr(g.ravel(), s.ravel())[0],
                         spearmanr(g.ravel(), t.ravel())[0]))
        if len(rows) >= args.n:
            break

rows = np.array(rows, dtype=object)
print(f'images: {len(rows)}  (median Spearman rho of G^t with each map; negative = higher prior when closer)')
for name in ('all', 'on-object', 'near', 'far'):
    r = rows if name == 'all' else rows[rows[:, 0] == name]
    if len(r) == 0:
        continue
    print(f'{name:10s} n={len(r):4d}  dist-to-fingertip {np.median(r[:, 1].astype(float)):+.2f}   '
          f'dist-to-ray-line {np.median(r[:, 2].astype(float)):+.2f}   along-ray t {np.median(r[:, 3].astype(float)):+.2f}')
