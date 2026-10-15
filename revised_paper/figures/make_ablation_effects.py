"""Ablation figure: what each component contributes, measured on the capability it was designed for.

Effect = full model minus ablation (points), or ablation minus full model for the direction error (degrees saved),
so positive always means "the component helps". 95% paired-bootstrap intervals over test images.
Run from the repository root: python paper/figures/make_ablation_effects.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from annotation import parse_annotation  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, 'runs')
INK, MUTED, POS, NEG, GRID, BAND = '#1f1f1f', '#8a8a8a', '#2A63B8', '#C4572A', '#e4e4e4', '#d6d6d6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': MUTED})
rng = np.random.default_rng(0)


def load(run):
    ev = glob.glob(os.path.join(RUNS, run, '*', 'evaluation'))[0]
    t = pd.read_csv(os.path.join(ev, 'predictions', 'test.csv')).set_index('name')
    mh = pd.read_csv(os.path.join(ev, 'predictions', 'test_multihand.csv')).set_index('name')
    return t, mh


full_t, full_mh = load('main_run')
area = pd.Series({n: (lambda b: b[2] * b[3])(parse_annotation(os.path.join(ROOT, 'EgoPoint_spatialnet', 'test', 'labels', n + '.txt'))['roi'])
                  for n in full_t[full_t.gt_T == 1].index})


def metric(t, mh, key):
    pos = t[t.gt_T == 1]
    if key == 'p50':
        return 100 * (pos.target_iou >= .5)
    if key == 'p70':
        return 100 * (pos.target_iou >= .7)
    if key == 'small':
        return 100 * (pos.loc[area[area < .03].index].target_iou >= .5)
    if key == 'far':
        return 100 * (pos[pos.gap_bin == 'far'].target_iou >= .5)
    if key == 'hand':
        return 100 * pos.hand_iou
    if key == 'dir':
        return pos.ray_err_deg
    if key == 'dir_far':
        return pos[pos.gap_bin == 'far'].ray_err_deg
    if key == 'others_p50':
        return 100 * (mh[mh.gt_T == 1].target_iou >= .5)
    if key == 'nohand_kept':
        return 100 * (mh[mh.gt_H == 0].pred_scene == 'No hand')
    raise KeyError(key)


def effect(run, key, lower_is_better=False):
    t, mh = load(run)
    a, b = metric(full_t, full_mh, key), metric(t, mh, key)
    idx = a.index.intersection(b.index)
    d = (b.loc[idx] - a.loc[idx]).astype(float).values if lower_is_better else (a.loc[idx] - b.loc[idx]).astype(float).values
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


SEEDS = {s: load(s) for s in ('seed_1', 'seed_2')}


def retrain_band(key):
    # Largest change that retraining the full model with another seed produced on this metric.
    base = metric(full_t, full_mh, key).mean()
    return max(abs(metric(*SEEDS[s], key).mean() - base) for s in SEEDS)


# (group, label, run, metric, lower_is_better)
POINTS = [
    ('A4 anchoring', 'P@0.7', 'A4_no_anchor_chain', 'p70', False),
    ('A4 anchoring', 'hand mIoU', 'A4_no_anchor_chain', 'hand', False),
    ('A4 anchoring', 'small targets, P@0.5', 'A4_no_anchor_chain', 'small', False),
    ('A4 anchoring', 'P@0.5', 'A4_no_anchor_chain', 'p50', False),
    ('A1 hand pasting', 'P@0.5, other hands in view', 'A1_no_copypaste', 'others_p50', False),
    ('A1 hand pasting', 'no-hand frames kept, other hands', 'A1_no_copypaste', 'nohand_kept', False),
    ('A1 hand pasting', 'P@0.5', 'A1_no_copypaste', 'p50', False),
    ('A2 consistency losses', 'far targets, P@0.5', 'A2_no_geometry_loss', 'far', False),
    ('A2 consistency losses', 'P@0.5', 'A2_no_geometry_loss', 'p50', False),
    ('A3 ray conditioning', 'far targets, P@0.5', 'A3_no_roi_geometry', 'far', False),
    ('A3 ray conditioning', 'P@0.5', 'A3_no_roi_geometry', 'p50', False),
]
DEGREES = [
    ('A4 anchoring', 'direction error', 'A4_no_anchor_chain', 'dir', True),
    ('A1 hand pasting', 'direction error', 'A1_no_copypaste', 'dir', True),
    ('A2 consistency losses', 'direction error', 'A2_no_geometry_loss', 'dir', True),
    ('A2 consistency losses', 'direction error, far', 'A2_no_geometry_loss', 'dir_far', True),
    ('A3 ray conditioning', 'direction error', 'A3_no_roi_geometry', 'dir', True),
]


SHORT = {('A4 anchoring', 'direction error'): 'A4 anchoring', ('A1 hand pasting', 'direction error'): 'A1 pasting',
         ('A2 consistency losses', 'direction error'): 'A2 losses, all',
         ('A2 consistency losses', 'direction error, far'): 'A2 losses, far',
         ('A3 ray conditioning', 'direction error'): 'A3 ray branch'}


def panel(ax, rows, unit, title, headers=True):
    groups = []
    y, ticks, labels = 0, [], []
    for g, lab, run, key, low in rows:
        if g not in groups and headers:
            if groups:
                y += 0.6
            groups.append(g)
            if headers:
                ax.text(-1.1, -y, g, transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
                        ha='left', va='center', fontsize=7.5, fontweight='bold', color=INK)
            y += 0.85
        m, lo, hi = effect(run, key, low)
        band = retrain_band(key)
        ax.add_patch(plt.Rectangle((-band, -y - 0.32), 2 * band, 0.64, color=BAND, lw=0, zorder=0))
        sig = lo > 0 or hi < 0
        col = (POS if m > 0 else NEG) if sig else MUTED
        ax.plot([lo, hi], [-y, -y], color=col, lw=1.6, solid_capstyle='round')
        ax.plot(m, -y, 'o', ms=4.5, color=col, mec='white', mew=0.8, zorder=3)
        ax.text(max(hi, band), -y, f'  {m:+.1f}', va='center', fontsize=6.6, color=INK if sig else MUTED)
        ticks.append(-y)
        labels.append(lab if headers else SHORT[(g, lab)])
        y += 1
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-y + 0.3, 0.6)
    lo_x, hi_x = ax.get_xlim()
    ax.set_xlim(lo_x, hi_x + 0.25 * (hi_x - lo_x))
    ax.set_xlabel(unit, fontsize=7)
    ax.set_title(title, fontsize=7.8, loc='left', color=INK)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)


fig = plt.figure(figsize=(5.5, 3.6))
ax1 = fig.add_axes([0.33, 0.12, 0.27, 0.80])
ax2 = fig.add_axes([0.80, 0.48, 0.17, 0.44])
panel(ax1, POINTS, 'gain from the component (points)', 'Localisation and robustness')
panel(ax2, DEGREES, 'degrees saved', 'Direction', headers=False)
ax2.text(-0.95, -0.42, 'Positive = the component helps.\n95% interval: blue above 0,\norange below 0, grey includes 0.\nShaded: change from retraining\nwith another seed.',
         transform=ax2.transAxes, fontsize=6.6, color=MUTED, va='top')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, 'ablation_effects.' + ext), dpi=300)
print('wrote', os.path.join(OUT, 'ablation_effects.pdf'))
