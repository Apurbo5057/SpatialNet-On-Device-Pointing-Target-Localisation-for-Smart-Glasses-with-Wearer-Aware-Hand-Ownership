"""Teaser figure (Fig. 1): SpatialNet predictions on four EgoPoint test frames.
Panel (d) uses the deterministic pasted-distractor test condition (test_evaluate.py --multihand).
Run from the repository root: python paper/figures/make_teaser.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from predict_util import DATA, predict, KeypointPointingDataset
from annotation import parse_annotation

OUT = os.path.dirname(os.path.abspath(__file__))
HAND, TIP, DIRC, TARGET, OTHER, GT = '#2A78D6', '#D2581F', '#4A3AA7', '#12805A', '#D62728', '#FFFFFF'
PANELS = [('img_0004948', '(a) fingertip on object'), ('img_0008587', '(b) object near'),
          ('img_0004634', '(c) object far'), ('img_0010638', '(d) other hands in view')]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})


def draw_box(ax, b, s, col, dashed=False, lw=1.6):
    ax.add_patch(patches.Rectangle(((b[0] - b[2] / 2) * s, (b[1] - b[3] / 2) * s), b[2] * s, b[3] * s,
                                   lw=lw, ec=col, fc='none', ls=(0, (3, 2)) if dashed else '-'))


fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.62))
for ax, (name, title) in zip(axes, PANELS):
    a = parse_annotation(os.path.join(DATA, 'test', 'labels', name + '.txt'))
    img = np.array(Image.open(os.path.join(DATA, 'test', 'images', name + '.jpg')).convert('RGB'))
    distractors = []
    if title.startswith('(d)'):
        mh = KeypointPointingDataset(DATA, 'test', augment=False, synthetic_multihand=True, seed=42)
        idx = next(i for i, smp in enumerate(mh.samples) if smp['name'] == name)
        img, distractors = mh._maybe_copy_paste(img.copy(), idx, mh.samples[idx], True, a['hand'], a['roi'], True)
    p = predict(img)
    s = img.shape[0]
    ax.imshow(img)
    draw_box(ax, a['roi'], s, GT, dashed=True, lw=1.0)
    for d in distractors:
        draw_box(ax, d, s, OTHER, dashed=True, lw=1.2)
    draw_box(ax, p['roi'], s, TARGET)
    draw_box(ax, p['hand'], s, HAND)
    tip = p['tip']
    ax.annotate('', xy=(tip + 0.16 * p['dir']) * s, xytext=tip * s,
                arrowprops=dict(arrowstyle='-|>', color=DIRC, lw=1.6, mutation_scale=9))
    ax.plot(tip[0] * s, tip[1] * s, 'o', ms=4.5, color=TIP, mec='white', mew=0.9)
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.set_xlim(0, s)
    ax.set_ylim(s, 0)
    ax.axis('off')

handles = [patches.Patch(ec=HAND, fc='none', lw=1.6, label="wearer's hand"),
           plt.Line2D([], [], marker='o', ls='', color=TIP, mec='white', ms=5, label='fingertip'),
           plt.Line2D([], [], color=DIRC, lw=1.6, label='direction'),
           patches.Patch(ec=TARGET, fc='none', lw=1.6, label='predicted target'),
           patches.Patch(ec='#888888', fc='none', lw=1.0, ls='--', label='labelled target'),
           patches.Patch(ec=OTHER, fc='none', lw=1.2, ls='--', label='pasted other hand')]
fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=6.2,
           handlelength=1.3, columnspacing=0.8, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.12, wspace=0.04)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, 'teaser.' + ext), dpi=300)
print('wrote', os.path.join(OUT, 'teaser.pdf'))
