"""Error breakdown of the target head for one or more runs (test split, pointing-at-target images).

    python paper/scripts/error_breakdown.py runs/main_run runs/img224

Per run: share of predicted centres inside the labelled box, oracle-size / oracle-centre P@0.5,
P@0.5 by target size, failure types (IoU < 0.3), and the efficiency rows of evaluation/metrics.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from annotation import parse_annotation  # noqa: E402

LABELS = os.path.join(ROOT, 'EgoPoint_spatialnet', 'test', 'labels')


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1, bx2, by2 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    i = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None) * np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    return i / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - i)


for run in sys.argv[1:]:
    ev = glob.glob(os.path.join(run, '*', 'evaluation'))[0]
    p = pd.read_csv(os.path.join(ev, 'predictions', 'test.csv'))
    p = p[p.gt_T == 1].reset_index(drop=True)
    g = np.array([parse_annotation(os.path.join(LABELS, n + '.txt'))['roi'] for n in p.name])
    pr = p[['pred_target_cx', 'pred_target_cy', 'pred_target_w', 'pred_target_h']].values
    inside = (np.abs(pr[:, 0] - g[:, 0]) <= g[:, 2] / 2) & (np.abs(pr[:, 1] - g[:, 1]) <= g[:, 3] / 2)
    t = p.target_iou.values
    area = g[:, 2] * g[:, 3]
    ratio = np.log(pr[:, 2] * pr[:, 3] / area)
    print(f'\n===== {os.path.basename(run.rstrip(chr(92) + "/"))}')
    print(f'centre inside labelled box: {100 * inside.mean():.1f}%   P@0.5 when inside: {100 * (t[inside] >= .5).mean():.1f}%')
    print(f'oracle size (own centre + true w,h): P@0.5 {100 * (iou(np.c_[pr[:, :2], g[:, 2:]], g) >= .5).mean():.1f}%')
    print(f'oracle centre (true centre + own w,h): P@0.5 {100 * (iou(np.c_[g[:, :2], pr[:, 2:]], g) >= .5).mean():.1f}%')
    print(f'boxes off by more than 2x in area: {100 * (np.abs(ratio) > np.log(2)).mean():.1f}%')
    for lo, hi in zip([0, .03, .08, .2, .45], [.03, .08, .2, .45, 1.01]):
        m = (area >= lo) & (area < hi)
        print(f'  target area {lo:.2f}-{hi:.2f}: n={m.sum():4d}  P@0.5 {100 * (t[m] >= .5).mean():5.1f}%  '
              f'area pred/true x{np.exp(np.median(ratio[m])):.2f}')
    f = t < 0.3
    miss = f & (p.pred_scene == 'No hand').values
    ext = f & ~miss & inside
    rest = f & ~miss & ~inside
    print(f'failures (IoU<0.3): {f.sum()} ({100 * f.mean():.1f}%)  '
          f'extent {ext.sum()}  direction>45deg {(rest & (p.ray_err_deg > 45).values).sum()}  '
          f'other object {(rest & (p.ray_err_deg <= 45).values).sum()}  hand missed {miss.sum()}')
    m = json.load(open(os.path.join(ev, 'metrics.json'), encoding='utf-8'))
    eff = m.get('efficiency', {})
    print('efficiency:', {k: eff[k] for k in eff if k in ('gflops', 'params_total', 'latency_ms', 'latency')})
