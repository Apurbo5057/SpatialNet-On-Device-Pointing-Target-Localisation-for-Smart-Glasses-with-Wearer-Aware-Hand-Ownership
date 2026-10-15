"""Qualitative successes and failures (Fig. 6) from the main run's test predictions.
Successes: IoU >= 0.7 per distance stratum. Failures: one per failure type (IoU < 0.3).
Run from the repository root: python paper/figures/make_qualitative.py [--candidates]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from predict_util import DATA, ROOT, predict
from annotation import parse_annotation

OUT = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(ROOT, 'runs', 'main_run', 'outputs_spatialnet', 'evaluation', 'predictions', 'test.csv')
HAND, TIP, DIRC, TARGET, GT = '#2A78D6', '#D2581F', '#4A3AA7', '#12805A', '#FFFFFF'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})

p = pd.read_csv(PRED)
p = p[p.gt_T == 1].copy()
lab = {n: parse_annotation(os.path.join(DATA, 'test', 'labels', n + '.txt')) for n in p.name}
g = np.array([lab[n]['roi'] for n in p.name])
p['centre_in'] = ((np.abs(p.pred_target_cx - g[:, 0]) <= g[:, 2] / 2) & (np.abs(p.pred_target_cy - g[:, 1]) <= g[:, 3] / 2)).values

GROUPS = {
    'ok on object': p[(p.gap_bin == 'on-object') & (p.target_iou >= 0.7)],
    'ok near': p[(p.gap_bin == 'near') & (p.target_iou >= 0.7)],
    'ok far': p[(p.gap_bin == 'far') & (p.target_iou >= 0.6)],
    'ok cluttered': p[(p.target_iou >= 0.7) & (p.tip_target_gap > 0.03)],
    'wrong object': p[(~p.centre_in) & (p.target_iou < 0.1) & (p.ray_err_deg < 20)],
    'wrong extent': p[p.centre_in & (p.target_iou < 0.3)],
    'wrong direction': p[(p.ray_err_deg > 60) & (p.target_iou < 0.1) & (p.gap_bin != 'on-object')],
    'hand missed': p[p.pred_scene == 'No hand'],
}


def draw(ax, name, title):
    img = np.array(Image.open(os.path.join(DATA, 'test', 'images', name + '.jpg')).convert('RGB'))
    a, q, S = lab[name], predict(img), img.shape[0]

    def box(b, col, dashed=False, lw=1.5):
        ax.add_patch(patches.Rectangle(((b[0] - b[2] / 2) * S, (b[1] - b[3] / 2) * S), b[2] * S, b[3] * S,
                                       lw=lw, ec=col, fc='none', ls=(0, (3, 2)) if dashed else '-'))
    ax.imshow(img)
    box(a['roi'], GT, dashed=True, lw=1.0)
    box(q['roi'], TARGET)
    box(q['hand'], HAND)
    ax.annotate('', xy=(q['tip'] + 0.16 * q['dir']) * S, xytext=q['tip'] * S,
                arrowprops=dict(arrowstyle='-|>', color=DIRC, lw=1.5, mutation_scale=8))
    ax.plot(q['tip'][0] * S, q['tip'][1] * S, 'o', ms=4, color=TIP, mec='white', mew=0.8)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlim(0, S)
    ax.set_ylim(S, 0)
    ax.axis('off')


if '--candidates' in sys.argv:
    fig, axes = plt.subplots(8, 6, figsize=(12, 17))
    for r, (k, df) in enumerate(GROUPS.items()):
        for c, n in enumerate(df.sample(min(6, len(df)), random_state=1).name):
            draw(axes[r, c], n, f'{k} {n} IoU {df.set_index("name").target_iou[n]:.2f}')
        for c in range(min(6, len(df)), 6):
            axes[r, c].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '_qual_candidates.png'), dpi=60)
    print({k: len(v) for k, v in GROUPS.items()})
    sys.exit(0)

CHOICE = {'on object': 'img_0002020', 'near': 'img_0015900', 'far': 'img_0005647', 'wide target': 'img_0010935',
          'wrong object': 'img_0001526', 'wrong extent': 'img_0003704', 'wrong direction': 'img_0006554',
          'hand missed': 'img_0006080'}  # chosen from --candidates (no faces, no watermarks)
fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.05))
for ax, (k, n) in zip(axes.flat, CHOICE.items()):
    iou = p.set_index('name').target_iou[n]
    draw(ax, n, f'{k} (IoU {iou:.2f})')
fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.04, hspace=0.14)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, 'qualitative.' + ext), dpi=300)
print('wrote', os.path.join(OUT, 'qualitative.pdf'))
