"""Teaser figure (Fig. 1): the task on four EgoPoint test frames, drawn from ground-truth labels.
Panel (d) uses the deterministic pasted-distractor test condition (test_evaluate.py --multihand).
Run from the repository root: python paper/figures/make_teaser.py
After the main run, replace the ground-truth overlays with predictions (predictions/test.csv).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from annotation import parse_annotation  # noqa: E402
from dataset import KeypointPointingDataset  # noqa: E402

DATA = os.path.join(ROOT, 'EgoPoint_spatialnet')
OUT = os.path.dirname(os.path.abspath(__file__))
HAND, TIP, DIRC, TARGET, OTHER = '#2A78D6', '#D2581F', '#4A3AA7', '#12805A', '#D62728'
PANELS = [('img_0004948', '(a) fingertip on object'), ('img_0008587', '(b) object near'),
          ('img_0004634', '(c) object far'), ('img_0010638', '(d) other hands in view')]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})


def draw_box(ax, b, s, col, dashed=False, lw=1.6):
    ax.add_patch(patches.Rectangle(((b[0] - b[2] / 2) * s, (b[1] - b[3] / 2) * s), b[2] * s, b[3] * s,
                                   lw=lw, ec=col, fc='none', ls=(0, (3, 2)) if dashed else '-'))


def draw_pointing(ax, a, s):
    tip, roi = np.array(a['fingertip']), np.array(a['roi'][:2])
    u = roi - tip
    if np.linalg.norm(u) > 1e-6:
        u = u / np.linalg.norm(u)
        L = max(0.12, min(0.35, np.linalg.norm(roi - tip)))
        ax.annotate('', xy=((tip + L * u) * s), xytext=(tip * s),
                    arrowprops=dict(arrowstyle='-|>', color=DIRC, lw=1.6, mutation_scale=9))
    ax.plot(tip[0] * s, tip[1] * s, 'o', ms=4.5, color=TIP, mec='white', mew=0.9)


fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.62))
test_mh = None
for ax, (name, title) in zip(axes, PANELS):
    a = parse_annotation(os.path.join(DATA, 'test', 'labels', name + '.txt'))
    img = np.array(Image.open(os.path.join(DATA, 'test', 'images', name + '.jpg')).convert('RGB'))
    distractors = []
    if title.startswith('(d)'):
        test_mh = KeypointPointingDataset(DATA, 'test', augment=False, synthetic_multihand=True, seed=42)
        idx = next(i for i, smp in enumerate(test_mh.samples) if smp['name'] == name)
        img, distractors = test_mh._maybe_copy_paste(img.copy(), idx, test_mh.samples[idx], True,
                                                     a['hand'], a['roi'], True)
    s = img.shape[0]
    ax.imshow(img)
    draw_box(ax, a['roi'], s, TARGET)
    draw_box(ax, a['hand'], s, HAND)
    for d in distractors:
        draw_box(ax, d, s, OTHER, dashed=True)
    draw_pointing(ax, a, s)
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.set_xlim(0, s)
    ax.set_ylim(s, 0)
    ax.axis('off')

handles = [patches.Patch(ec=HAND, fc='none', lw=1.6, label="wearer's hand"),
           plt.Line2D([], [], marker='o', ls='', color=TIP, mec='white', ms=5, label='fingertip'),
           plt.Line2D([], [], color=DIRC, lw=1.6, label='pointing direction'),
           patches.Patch(ec=TARGET, fc='none', lw=1.6, label='pointed-at object'),
           patches.Patch(ec=OTHER, fc='none', lw=1.2, ls='--', label='other hand (ignore)')]
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=6.6,
           handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.12, wspace=0.04)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, 'teaser.' + ext), dpi=300)
print('wrote', os.path.join(OUT, 'teaser.pdf'))
