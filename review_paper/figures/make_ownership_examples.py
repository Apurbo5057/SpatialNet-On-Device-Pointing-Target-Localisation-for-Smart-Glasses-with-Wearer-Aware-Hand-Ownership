"""Wearer-ownership augmentation examples (Fig. 3), made with the training code's own
copy-paste function (dataset.copy_paste_distractors), so the figure shows exactly what training sees.
Run from the repository root: python paper/figures/make_ownership_examples.py [--candidates]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dataset import KeypointPointingDataset, copy_paste_distractors, _crop_box  # noqa: E402

DATA = os.path.join(ROOT, 'EgoPoint_spatialnet')
OUT = os.path.dirname(os.path.abspath(__file__))
HAND, TARGET, OTHER = '#2A78D6', '#12805A', '#D62728'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})

ds = KeypointPointingDataset(DATA, 'train', augment=False)
pointing = [i for i, s in enumerate(ds.samples) if s['flags'][2]]
nohand = [i for i, s in enumerate(ds.samples) if not s['flags'][0]]


def make(kind, seed):
    """kind: 'other1' | 'other2' | 'self' | 'nohand'. Returns (img, sample, distractor boxes) or None."""
    rng = np.random.RandomState(seed)
    base = int((nohand if kind == 'nohand' else pointing)[rng.randint(len(nohand if kind == 'nohand' else pointing))])
    s = ds.samples[base]
    img = ds._load_rgb(base)
    avoid = [] if kind == 'nohand' else [s['hand_bbox'], s['roi_bbox']]
    if kind == 'self':
        crops, num = [_crop_box(img, s['hand_bbox'])], 1
    else:
        num = 2 if kind == 'other2' else 1
        crops = []
        for _ in range(num):
            j = int(pointing[rng.randint(len(pointing))])
            crops.append(_crop_box(ds._load_rgb(j), ds.samples[j]['hand_bbox']))
    img, boxes = copy_paste_distractors(np.ascontiguousarray(img.copy()), avoid, crops, rng, num)
    if len(boxes) < num:
        return None
    return img, s, boxes


def draw(ax, img, s, boxes, title):
    S = img.shape[0]
    ax.imshow(img)

    def box(b, col, dashed=False):
        ax.add_patch(patches.Rectangle(((b[0] - b[2] / 2) * S, (b[1] - b[3] / 2) * S), b[2] * S, b[3] * S,
                                       lw=1.6, ec=col, fc='none', ls=(0, (3, 2)) if dashed else '-'))
    if s['flags'][0]:
        box(s['roi_bbox'], TARGET)
        box(s['hand_bbox'], HAND)
    for b in boxes:
        box(b, OTHER, dashed=True)
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.set_xlim(0, S)
    ax.set_ylim(S, 0)
    ax.axis('off')


if '--candidates' in sys.argv:
    kinds = ['other1', 'other2', 'self', 'nohand']
    fig, axes = plt.subplots(4, 6, figsize=(14, 9.5))
    for r, k in enumerate(kinds):
        seed, c = 0, 0
        while c < 6:
            out = make(k, seed)
            if out is not None:
                draw(axes[r, c], *out, f'{k} seed={seed}')
                c += 1
            seed += 1
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '_ownership_candidates.png'), dpi=70)
    sys.exit(0)

CHOICE = [('other1', 1, '(a) one other hand'), ('other2', 4, '(b) two other hands'),
          ('self', 3, '(c) own hand re-pasted'), ('nohand', 3, '(d) no wearer hand, H=0')]
fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.62))
for ax, (k, seed, title) in zip(axes, CHOICE):
    out = make(k, seed)
    draw(ax, *out, title)
handles = [patches.Patch(ec=HAND, fc='none', lw=1.6, label="wearer's hand (label)"),
           patches.Patch(ec=TARGET, fc='none', lw=1.6, label='target (label)'),
           patches.Patch(ec=OTHER, fc='none', lw=1.2, ls='--', label='pasted hand (no label; extra heatmap penalty)')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=6.6,
           handlelength=1.4, columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.12, wspace=0.04)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, 'ownership_examples.' + ext), dpi=300)
print('wrote', os.path.join(OUT, 'ownership_examples.pdf'))
