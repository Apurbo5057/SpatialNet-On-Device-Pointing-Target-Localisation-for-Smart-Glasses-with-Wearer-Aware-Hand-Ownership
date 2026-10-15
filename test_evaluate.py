"""
Evaluate a SpatialNet checkpoint on the test split and save every statistic, table and figure the
paper needs, plus per-image predictions so any number can be recomputed without re-running.
Usage: python test_evaluate.py --model_path <out>/best_model.pth --data_dir <dataset> --output_dir <out>/evaluation
                               [--multihand] [--calibrate_thresholds]
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch

from model import SpatialNet
from dataset import get_eval_loader, get_multihand_loader, gap_bin
import config as cfg
import reporting as R
from train import add_model_args

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# scene ids returned by derive_pred_category, and their H P T flags
CAT_NAMES = {0: 'no_hand', 1: 'non_pointing', 2: 'pointing', 3: 'pt_empty'}
CAT_FLAGS = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 1), 3: (1, 1, 0)}
FLAGS_CAT = {v: k for k, v in CAT_FLAGS.items()}
ORDER = [FLAGS_CAT[f] for f in R.SCENES]                 # no hand, non-pointing, pointing no target, at target
NAME = {c: R.SCENE_NAMES[CAT_FLAGS[c]] for c in CAT_FLAGS}
IOU_THRESHOLDS = (0.3, 0.5, 0.7, 0.75)
GAP_BINS = ['on-object', 'near', 'far']


def derive_pred_category(hand_conf, point_conf, roi_conf=1.0,
                         hc_thresh=0.5, pc_thresh=0.5, rc_thresh=0.5):
    """Cascade: no hand -> non-pointing -> pointing-empty -> pointing."""
    if hand_conf < hc_thresh:
        return 0
    if point_conf < pc_thresh:
        return 1
    if roi_conf < rc_thresh:
        return 3
    return 2


def load_model(model_path, args, device):
    """Rebuild the exact architecture the checkpoint was trained with."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False) if os.path.exists(model_path) else None
    saved = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    model = SpatialNet(
        input_size=saved.get('img_size', args.img_size), load_pretrained=False,
        use_bifpn=not saved.get('no_bifpn', args.no_bifpn),
        use_coordconv=not saved.get('no_coordconv', args.no_coordconv),
        mixer_depth=saved.get('mixer_depth', args.mixer_depth),
        anchor_chain=not saved.get('no_anchor_chain', args.no_anchor_chain),
        local_softargmax=not saved.get('global_softargmax', args.global_softargmax),
        roi_geometry=saved.get('roi_geometry', args.roi_geometry),
        ft_box_prior=not saved.get('no_ft_box_prior', args.no_ft_box_prior),
        roi_forward_only=saved.get('roi_forward_only', args.roi_forward_only),
        roi_size_step=saved.get('roi_size_step', args.roi_size_step))
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint at {model_path}')
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def checkpoint_info(model_path):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        return {}
    return {k: ckpt[k] for k in ('epoch', 'val_loss', 'selection_tier') if k in ckpt}


# geometry

def _corners(b):
    cx, cy, w, h = b
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = _corners(a)
    bx1, by1, bx2, by2 = _corners(b)
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return float(inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-8))


def point_in_box(p, box):
    x1, y1, x2, y2 = _corners(box)
    return bool(x1 <= p[0] <= x2 and y1 <= p[1] <= y2)


# predictions

@torch.no_grad()
def collect_predictions(model, loader, device, img_size, thresholds=(0.5, 0.5, 0.5)):
    """One record per image: ground truth, raw gate probabilities, predicted scene and per-task errors."""
    model.eval()
    names = [s['name'] for s in loader.dataset.samples]
    out = []
    for batch in loader:
        preds = model(batch['image'].to(device))
        probs = {k: torch.sigmoid(preds[k]).float().cpu().numpy().reshape(-1)
                 for k in ('hand_conf', 'point_conf', 'roi_conf')}
        p = {k: preds[k].float().cpu().numpy() for k in ('hand', 'fingertip', 'angle', 'roi')}
        for i in range(batch['image'].shape[0]):
            flags = tuple(int(batch[k][i].item() > 0.5) for k in ('hand_conf', 'point_conf', 'roi_conf'))
            hc, pc, rc = (float(probs[k][i]) for k in ('hand_conf', 'point_conf', 'roi_conf'))
            nd = int(batch['num_distractors'][i].item())
            gt_hand, gt_tip = batch['hand_bbox'][i].numpy(), batch['fingertip_kp'][i].numpy()
            gt_roi, gt_ang = batch['roi_bbox'][i].numpy(), batch['sin_cos_angle'][i].numpy()
            dist = batch['distractor_boxes'][i][:nd].numpy()
            s = {'name': names[int(batch['index'][i].item())], 'gt_flags': flags,
                 'gt_scene': FLAGS_CAT[flags], 'hand_prob': hc, 'pointing_prob': pc, 'target_prob': rc,
                 'pred_scene': derive_pred_category(hc, pc, rc, *thresholds),
                 'gap': float(batch['ft_roi_gap'][i].item()), 'n_distractors': nd,
                 'pred_hand': p['hand'][i], 'pred_fingertip': p['fingertip'][i],
                 'pred_angle': p['angle'][i], 'pred_roi': p['roi'][i], 'gt_hand': gt_hand,
                 'gt_fingertip': gt_tip, 'gt_roi': gt_roi, 'distractors': dist,
                 'hand_iou': None, 'target_iou': None, 'tip_err_px': None, 'ray_err_deg': None,
                 'owner_correct': None, 'pred_on_distractor': None}
            if flags[0]:
                s['hand_iou'] = box_iou(p['hand'][i], gt_hand)
            if flags[1]:
                s['tip_err_px'] = float(np.linalg.norm((p['fingertip'][i] - gt_tip) * img_size))
                c = float(np.clip(np.dot(p['angle'][i], gt_ang), -1.0, 1.0))
                s['ray_err_deg'] = float(np.degrees(np.arccos(c)))
            if flags[2]:
                s['target_iou'] = box_iou(p['roi'][i], gt_roi)
            if nd > 0 and flags[0]:
                c = p['hand'][i][:2]
                s['owner_correct'] = bool(np.linalg.norm(c - gt_hand[:2]) <
                                          np.min(np.linalg.norm(dist[:, :2] - c, axis=1)))
                s['pred_on_distractor'] = any(point_in_box(c, b) for b in dist)
            out.append(s)
    return out


def rescore(samples, thresholds):
    """Copy of `samples` with the predicted scene recomputed at other gate thresholds."""
    return [{**s, 'pred_scene': derive_pred_category(s['hand_prob'], s['pointing_prob'], s['target_prob'],
                                                     *thresholds)} for s in samples]


# metrics

def iou_stats(values):
    a = np.asarray([v for v in values if v is not None], float)
    if a.size == 0:
        return {'n': 0}
    lo, hi = R.bootstrap_ci(a)
    out = {'n': int(a.size), 'mean': float(a.mean()), 'mean_ci': [lo, hi], 'median': float(np.median(a))}
    for t in IOU_THRESHOLDS:
        out[f'@{t:g}'] = float((a >= t).mean())
    return out


def err_stats(values, cuts):
    a = np.asarray([v for v in values if v is not None], float)
    if a.size == 0:
        return {'n': 0}
    lo, hi = R.bootstrap_ci(a)
    out = {'n': int(a.size), 'mean': float(a.mean()), 'mean_ci': [lo, hi], 'median': float(np.median(a)),
           'p90': float(np.percentile(a, 90))}
    for c in cuts:
        out[f'<{c:g}'] = float((a < c).mean())
    return out


def classification_metrics(samples):
    k = len(ORDER)
    pos = {c: i for i, c in enumerate(ORDER)}
    cm = np.zeros((k, k), dtype=int)
    for s in samples:
        cm[pos[s['gt_scene']], pos[s['pred_scene']]] += 1
    n = int(cm.sum())
    correct = int(np.trace(cm))
    per = {}
    for c in ORDER:
        i = pos[c]
        tp, sup, predn = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        prec = tp / predn if predn else None
        rec = tp / sup if sup else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if sup else None)
        per[NAME[c]] = {'support': sup, 'tp': tp, 'fp': predn - tp, 'fn': sup - tp, 'precision': prec,
                        'recall': rec, 'recall_ci': list(R.wilson(tp, sup)), 'f1': f1}
    f1s = [v['f1'] for v in per.values() if v['f1'] is not None]
    recs = [v['recall'] for v in per.values() if v['recall'] is not None]
    return {'n': n, 'accuracy': correct / n if n else None, 'accuracy_ci': list(R.wilson(correct, n)),
            'macro_f1': float(np.mean(f1s)) if f1s else None,
            'balanced_accuracy': float(np.mean(recs)) if recs else None,
            'per_scene': per, 'confusion': cm, 'labels': [NAME[c] for c in ORDER]}


def gate_metrics(samples, thresholds):
    """Each gate on its own population: hand on all images, pointing on hand images, target on pointing images."""
    gates = [('Hand present', 'hand_prob', 0, lambda s: True, thresholds[0]),
             ('Pointing', 'pointing_prob', 1, lambda s: s['gt_flags'][0] == 1, thresholds[1]),
             ('Target present', 'target_prob', 2, lambda s: s['gt_flags'][1] == 1, thresholds[2])]
    out = {}
    for name, key, fi, pop, t in gates:
        sub = [s for s in samples if pop(s)]
        y = np.array([s['gt_flags'][fi] for s in sub], bool)
        p = np.array([s[key] for s in sub], float)
        pred = p >= t
        out[name] = {'population': len(sub), 'positives': int(y.sum()), 'threshold': t,
                     'accuracy': float((pred == y).mean()) if len(sub) else None,
                     'missed': int((~pred & y).sum()), 'false_alarms': int((pred & ~y).sum()),
                     'recall': float(pred[y].mean()) if y.any() else None,
                     'auroc': R.auroc(y, p), 'ap': R.average_precision(y, p)}
    return out


def localisation_metrics(samples):
    by = lambda c: [s for s in samples if s['gt_scene'] == c]
    pt, pe, npt = by(FLAGS_CAT[(1, 1, 1)]), by(FLAGS_CAT[(1, 1, 0)]), by(FLAGS_CAT[(1, 0, 0)])
    hand_rows = {NAME[FLAGS_CAT[(1, 0, 0)]]: npt, NAME[FLAGS_CAT[(1, 1, 0)]]: pe,
                 NAME[FLAGS_CAT[(1, 1, 1)]]: pt, 'All hand-present': npt + pe + pt}
    tip_rows = {NAME[FLAGS_CAT[(1, 1, 0)]]: pe, NAME[FLAGS_CAT[(1, 1, 1)]]: pt, 'All pointing': pe + pt}
    target_rows = {'All': pt}
    for b in GAP_BINS:
        target_rows[f'Gap {b}'] = [s for s in pt if gap_bin(s['gap']) == b]
    return {
        'hand_iou': {k: iou_stats([s['hand_iou'] for s in v]) for k, v in hand_rows.items()},
        'target_iou': {k: iou_stats([s['target_iou'] for s in v]) for k, v in target_rows.items()},
        'fingertip_px': {k: err_stats([s['tip_err_px'] for s in v], (5, 10)) for k, v in tip_rows.items()},
        'ray_deg': {k: err_stats([s['ray_err_deg'] for s in v], (15, 30)) for k, v in tip_rows.items()},
        'target_by_gap_detail': {b: {
            'n': len(target_rows[f'Gap {b}']),
            'fingertip_px': err_stats([s['tip_err_px'] for s in target_rows[f'Gap {b}']], (5, 10)),
            'ray_deg': err_stats([s['ray_err_deg'] for s in target_rows[f'Gap {b}']], (15, 30)),
            'missed_pointing': sum(s['pred_scene'] not in (2, 3) for s in target_rows[f'Gap {b}'])}
            for b in GAP_BINS},
    }


def ownership_metrics(samples):
    own = [s for s in samples if s['owner_correct'] is not None]
    pt = [s for s in own if s['gt_scene'] == 2]
    nohand = [s for s in samples if s['gt_scene'] == 0 and s['n_distractors'] > 0]
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {'hand_images_with_distractors': len(own),
            'ownership_accuracy': mean([s['owner_correct'] for s in own]),
            'ownership_accuracy_ci': list(R.wilson(sum(s['owner_correct'] for s in own), len(own))),
            'hand_centre_on_distractor_rate': mean([s['pred_on_distractor'] for s in own]),
            'pointing_images_with_distractors': len(pt),
            'pointing_ownership_accuracy': mean([s['owner_correct'] for s in pt]),
            'no_hand_images_with_distractors': len(nohand),
            'no_hand_still_no_hand_rate': mean([s['pred_scene'] == 0 for s in nohand])}


def all_metrics(samples, thresholds, with_ownership=False):
    m = {'n': len(samples), 'thresholds': {'hand': thresholds[0], 'pointing': thresholds[1],
                                           'target': thresholds[2]},
         'classification': classification_metrics(samples), 'gates': gate_metrics(samples, thresholds),
         'localisation': localisation_metrics(samples)}
    if with_ownership:
        m['ownership'] = ownership_metrics(samples)
    return m


def calibrate_thresholds(val_samples, search_min=0.0, search_max=0.5, step=0.01):
    """Highest per-gate threshold on VAL with zero missed positives, in cascade order."""
    grid = np.arange(search_max, search_min - step / 2, -step)

    def best(key, fi, pop):
        sub = [s for s in val_samples if pop(s)]
        p = np.array([s[key] for s in sub])
        y = np.array([s['gt_flags'][fi] for s in sub], bool)
        for t in grid:
            if p.size == 0 or int(((p < t) & y).sum()) == 0:
                return float(round(t, 4))
        return search_min

    return (best('hand_prob', 0, lambda s: True),
            best('pointing_prob', 1, lambda s: s['gt_flags'][0] == 1),
            best('target_prob', 2, lambda s: s['gt_flags'][1] == 1))


# tables (headers, rows) shared by console, CSV and markdown

def _ci(st, scale=100, digits=1):
    if not st or st.get('n', 0) == 0 or st.get('mean_ci', [None])[0] is None:
        return '-'
    return f'{st["mean_ci"][0] * scale:.{digits}f}-{st["mean_ci"][1] * scale:.{digits}f}'


def table_classification(m):
    c = m['classification']
    rows = [[k, v['support'], R.pct(v['precision']), R.pct(v['recall']),
             '-' if v['recall_ci'][0] is None else f'{100 * v["recall_ci"][0]:.1f}-{100 * v["recall_ci"][1]:.1f}',
             R.pct(v['f1'])] for k, v in c['per_scene'].items()]
    return ['Scene (H P T)', 'Support', 'Precision %', 'Recall %', 'Recall 95% CI', 'F1 %'], rows


def table_confusion(m):
    c = m['classification']
    return ['Ground truth \\ Predicted'] + c['labels'], [[lab] + list(map(int, row))
                                                         for lab, row in zip(c['labels'], c['confusion'])]


def table_gates(m):
    rows = [[k, v['population'], v['positives'], f'{v["threshold"]:.2f}', R.pct(v['accuracy']),
             R.pct(v['recall']), v['missed'], v['false_alarms'],
             '-' if v['auroc'] is None else f'{v["auroc"]:.4f}', '-' if v['ap'] is None else f'{v["ap"]:.4f}']
            for k, v in m['gates'].items()]
    return ['Gate', 'Images', 'Positives', 'Threshold', 'Accuracy %', 'Recall %', 'Missed',
            'False alarms', 'AUROC', 'AP'], rows


def table_iou(block):
    rows = []
    for k, st in block.items():
        if st.get('n', 0) == 0:
            rows.append([k, 0] + ['-'] * (3 + len(IOU_THRESHOLDS)))
            continue
        rows.append([k, st['n'], R.pct(st['mean']), _ci(st), R.pct(st['median'])]
                    + [R.pct(st[f'@{t:g}']) for t in IOU_THRESHOLDS])
    return ['Subset', 'n', 'Mean IoU %', 'Mean 95% CI', 'Median %'] + \
        [f'IoU>={t:g} %' for t in IOU_THRESHOLDS], rows


def table_err(block, unit, cuts):
    rows = []
    for k, st in block.items():
        if st.get('n', 0) == 0:
            rows.append([k, 0] + ['-'] * (4 + len(cuts)))
            continue
        rows.append([k, st['n'], f'{st["mean"]:.2f}', _ci(st, 1, 2), f'{st["median"]:.2f}', f'{st["p90"]:.2f}']
                    + [R.pct(st[f'<{c:g}']) for c in cuts])
    return ['Subset', 'n', f'Mean {unit}', 'Mean 95% CI', f'Median {unit}', f'P90 {unit}'] + \
        [f'<{c:g} {unit} %' for c in cuts], rows


def table_compare(named_metrics):
    """Headline metrics side by side for several conditions."""
    def g(m, *path):
        for p in path:
            m = m.get(p) if isinstance(m, dict) else None
        return m
    rows_def = [
        ('Scene accuracy %', lambda m: R.pct(g(m, 'classification', 'accuracy'))),
        ('Scene macro-F1 %', lambda m: R.pct(g(m, 'classification', 'macro_f1'))),
        ('Missed hand / pointing / target', lambda m: '/'.join(str(g(m, 'gates', k, 'missed'))
                                                               for k in ('Hand present', 'Pointing', 'Target present'))),
        ('False alarms hand / pointing / target', lambda m: '/'.join(str(g(m, 'gates', k, 'false_alarms'))
                                                                     for k in ('Hand present', 'Pointing', 'Target present'))),
        ('Hand IoU mean % (all hand-present)', lambda m: R.pct(g(m, 'localisation', 'hand_iou', 'All hand-present', 'mean'))),
        ('Hand IoU>=0.5 %', lambda m: R.pct(g(m, 'localisation', 'hand_iou', 'All hand-present', '@0.5'))),
        ('Target IoU mean %', lambda m: R.pct(g(m, 'localisation', 'target_iou', 'All', 'mean'))),
        ('Target IoU>=0.5 %', lambda m: R.pct(g(m, 'localisation', 'target_iou', 'All', '@0.5'))),
        ('Fingertip error mean px', lambda m: f'{g(m, "localisation", "fingertip_px", "All pointing", "mean") or 0:.2f}'),
    ]
    names = list(named_metrics)
    rows = [[label] + [fn(named_metrics[n]) for n in names] for label, fn in rows_def]
    return ['Metric'] + names, rows


# figures

def fig_confusion(m, path):
    plt = R.paper_style()
    from matplotlib.colors import LinearSegmentedColormap
    c = m['classification']
    cm = c['confusion'].astype(float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.grid(False)
    ax.imshow(norm, cmap=LinearSegmentedColormap.from_list('seq', R.SEQ_BLUE), vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label = f'{100 * norm[i, j]:.1f}%\n({int(cm[i, j])})' if cm[i, j] else '0'
            ax.text(j, i, label, ha='center', va='center', fontsize=6.5,
                    color='white' if norm[i, j] > 0.55 else (R.INK if cm[i, j] else R.MUTED))
    labels = [l.replace(', ', ',\n').replace(' at ', '\nat ') for l in c['labels']]
    ax.set_xticks(range(len(labels)), labels, fontsize=6.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground truth')
    ax.set_title(f'Scene classification, accuracy {100 * c["accuracy"]:.1f}%', loc='left')
    R.save_figure(fig, path)


def fig_gate_curves(samples, m, path_roc, path_pr):
    plt = R.paper_style()
    spec = [('Hand present', 'hand_prob', 0, lambda s: True),
            ('Pointing', 'pointing_prob', 1, lambda s: s['gt_flags'][0] == 1),
            ('Target present', 'target_prob', 2, lambda s: s['gt_flags'][1] == 1)]
    curves = []
    for (name, key, fi, pop), col, mk in zip(spec, [R.C1, R.C2, R.C3], ['o', 's', '^']):
        sub = [s for s in samples if pop(s)]
        y = np.array([s['gt_flags'][fi] for s in sub], bool)
        if y.all() or not y.any():
            continue
        curves.append((name, R.roc_pr_curves(y, [s[key] for s in sub]), col, mk))
    for kind, path in (('roc', path_roc), ('pr', path_pr)):
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        for name, cv, col, mk in curves:
            g = m['gates'][name]
            if kind == 'roc':
                ax.plot(cv['fpr'], cv['tpr'], color=col, marker=mk, markevery=50,
                        label=f'{name} (AUROC {g["auroc"]:.3f})')
            else:
                ax.plot(cv['recall'], cv['precision'], color=col, marker=mk, markevery=50,
                        label=f'{name} (AP {g["ap"]:.3f})')
        if kind == 'roc':
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title('Gate ROC curves', loc='left')
        else:
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Gate precision-recall curves', loc='left')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc='lower right' if kind == 'roc' else 'lower left')
        R.save_figure(fig, path)


def fig_iou_success(samples, path):
    plt = R.paper_style()
    t = np.linspace(0, 1, 101)
    hand = np.array([s['hand_iou'] for s in samples if s['hand_iou'] is not None])
    tgt = np.array([s['target_iou'] for s in samples if s['target_iou'] is not None])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for a, name, col, mk in ((hand, 'Hand (all hand-present)', R.C1, 'o'), (tgt, 'Target (pointing at target)', R.C2, 's')):
        if a.size:
            ax.plot(t, [(a >= x).mean() for x in t], color=col, marker=mk, markevery=10, label=name)
    for x in (0.3, 0.5, 0.7):
        ax.axvline(x, color=R.GRID, lw=0.8, zorder=0)
    ax.set_xlabel('IoU threshold')
    ax.set_ylabel('Fraction of images')
    ax.set_title('Localisation success rate vs IoU threshold', loc='left')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower left')
    R.save_figure(fig, path)


def fig_error_cdf(samples, key, xlabel, title, path):
    plt = R.paper_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for scene, col, mk in ((FLAGS_CAT[(1, 1, 1)], R.C1, 'o'), (FLAGS_CAT[(1, 1, 0)], R.C2, 's')):
        a = np.sort([s[key] for s in samples if s['gt_scene'] == scene and s[key] is not None])
        if a.size:
            ax.plot(a, np.arange(1, a.size + 1) / a.size, color=col, marker=mk,
                    markevery=max(1, a.size // 10), label=f'{NAME[scene]} (n={a.size})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative fraction')
    ax.set_title(title, loc='left')
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc='lower right')
    R.save_figure(fig, path)


def fig_target_by_gap(m, path):
    plt = R.paper_style()
    blk = m['localisation']['target_iou']
    labels = [b for b in GAP_BINS if blk[f'Gap {b}'].get('n', 0)]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(labels))
    w = 0.38
    mean = [100 * blk[f'Gap {b}']['mean'] for b in labels]
    at50 = [100 * blk[f'Gap {b}']['@0.5'] for b in labels]
    b1 = ax.bar(x - w / 2 - 0.01, mean, w, color=R.C1, label='Mean IoU')
    b2 = ax.bar(x + w / 2 + 0.01, at50, w, color=R.C2, label='IoU>=0.5')
    ax.bar_label(b1, fmt='%.0f', fontsize=6.5, padding=2, color=R.INK2)
    ax.bar_label(b2, fmt='%.0f', fontsize=6.5, padding=2, color=R.INK2)
    ax.set_xticks(x, [f'{b}\n(n={blk[f"Gap {b}"]["n"]})' for b in labels])
    ax.set_ylabel('%')
    ax.set_ylim(0, 110)
    ax.set_title('Target localisation by fingertip-target gap', loc='left')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='x', visible=False)
    R.save_figure(fig, path)


def fig_qualitative(samples, dataset, img_size, path, per_scene=4, seed=0):
    plt = R.paper_style()
    import matplotlib.patches as patches
    rng = random.Random(seed)
    picked = []
    for c in reversed(ORDER):
        pool = [i for i, s in enumerate(samples) if s['gt_scene'] == c]
        picked += rng.sample(pool, min(per_scene, len(pool)))
    if not picked:
        return
    cols = 4
    rows = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows))
    axes = np.array(axes).reshape(-1)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def rect(ax, b, col, ls):
        x1, y1, x2, y2 = (v * img_size for v in _corners(b))
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.2, ec=col, fc='none', ls=ls))

    for ax, i in zip(axes, picked):
        s = samples[i]
        img = dataset[[k for k, x in enumerate(dataset.samples) if x['name'] == s['name']][0]]['image']
        ax.imshow(np.clip(img.permute(1, 2, 0).numpy() * std + mean, 0, 1))
        ax.grid(False)
        g, p = s['gt_flags'], CAT_FLAGS[s['pred_scene']]
        if g[0]:
            rect(ax, s['gt_hand'], 'white', '--')
        if g[2]:
            rect(ax, s['gt_roi'], 'white', '--')
        if p[0]:
            rect(ax, s['pred_hand'], R.C1, '-')
        if p[1]:
            ft = s['pred_fingertip'] * img_size
            ax.plot(*ft, 'o', color=R.C2, ms=4, mec='white', mew=0.6)
            ax.annotate('', xy=(ft[0] + s['pred_angle'][1] * 30, ft[1] + s['pred_angle'][0] * 30), xytext=ft,
                        arrowprops=dict(arrowstyle='->', color=R.C2, lw=1.2))
        if p[2]:
            rect(ax, s['pred_roi'], R.C3, '-')
        ax.set_xlim(0, img_size)
        ax.set_ylim(img_size, 0)
        ok = s['gt_scene'] == s['pred_scene']
        ax.set_title(f'GT {"".join(map(str, g))}  pred {"".join(map(str, p))}',
                     fontsize=7, color=R.INK if ok else '#d03b3b')
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(picked):]:
        ax.axis('off')
    fig.suptitle('Dashed white: ground truth. Blue: hand, orange: fingertip and ray, aqua: target. '
                 'Titles give H P T flags (red = wrong scene).', fontsize=7, color=R.INK2, x=0.01, ha='left')
    fig.tight_layout(h_pad=1.5)
    fig.savefig(path + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# outputs

PRED_COLUMNS = ['name', 'gt_H', 'gt_P', 'gt_T', 'gt_scene', 'pred_scene', 'correct', 'hand_prob',
                'pointing_prob', 'target_prob', 'hand_iou', 'target_iou', 'tip_err_px', 'ray_err_deg',
                'tip_target_gap', 'gap_bin', 'n_distractors', 'owner_correct', 'pred_on_distractor',
                'pred_hand_cx', 'pred_hand_cy', 'pred_hand_w', 'pred_hand_h', 'pred_tip_x', 'pred_tip_y',
                'pred_target_cx', 'pred_target_cy', 'pred_target_w', 'pred_target_h']


def save_predictions(samples, path):
    r6 = lambda v: None if v is None else round(float(v), 6)
    rows = []
    for s in samples:
        rows.append([s['name'], *s['gt_flags'], NAME[s['gt_scene']], NAME[s['pred_scene']],
                     int(s['gt_scene'] == s['pred_scene']), r6(s['hand_prob']), r6(s['pointing_prob']),
                     r6(s['target_prob']), r6(s['hand_iou']), r6(s['target_iou']), r6(s['tip_err_px']),
                     r6(s['ray_err_deg']), r6(s['gap']) if s['gap'] >= 0 else None,
                     gap_bin(s['gap']) if s['gap'] >= 0 else None, s['n_distractors'],
                     s['owner_correct'], s['pred_on_distractor'],
                     *[r6(v) for v in s['pred_hand']], *[r6(v) for v in s['pred_fingertip']],
                     *[r6(v) for v in s['pred_roi']]])
    R.write_csv(path, PRED_COLUMNS, rows)


class Report:
    """Prints each table and collects it for tables/*.csv and summary.md."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.md = ['# SpatialNet evaluation', '']

    def heading(self, title):
        R.section(title)
        self.md += [f'## {title}', '']

    def note(self, text):
        print(f'  {text}')
        self.md += [text, '']

    def table(self, name, headers, rows):
        R.print_table(headers, rows)
        print()
        R.write_csv(os.path.join(self.out_dir, 'tables', f'{name}.csv'), headers, rows)
        self.md += [R.markdown_table(headers, rows), '']

    def save(self):
        with open(os.path.join(self.out_dir, 'summary.md'), 'w') as f:
            f.write('\n'.join(self.md))


def report_condition(rep, m, tag):
    c = m['classification']
    rep.note(f'{c["n"]} images. Scene accuracy {R.pct(c["accuracy"], 2)}% '
             f'(95% CI {R.pct(c["accuracy_ci"][0])}-{R.pct(c["accuracy_ci"][1])}), '
             f'macro-F1 {R.pct(c["macro_f1"], 2)}%, balanced accuracy {R.pct(c["balanced_accuracy"], 2)}%. '
             f'Gate thresholds hand/pointing/target = '
             f'{m["thresholds"]["hand"]:.2f}/{m["thresholds"]["pointing"]:.2f}/{m["thresholds"]["target"]:.2f}.')
    rep.table(f'{tag}_scene_classification', *table_classification(m))
    rep.table(f'{tag}_confusion_matrix', *table_confusion(m))
    rep.table(f'{tag}_gates', *table_gates(m))


def run_evaluation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    out = args.output_dir
    for d in ('figures', 'tables', 'predictions'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    t_start = time.time()

    R.banner('SpatialNet evaluation')
    model = load_model(args.model_path, args, device)
    img_size = model.input_size if hasattr(model, 'input_size') else args.img_size
    ck = checkpoint_info(args.model_path)
    thresholds = (args.hc_thresh, args.pc_thresh, args.rc_thresh)
    test_dl = get_eval_loader(args.data_dir, 'test', img_size, args.batch_size)
    env = R.env_info(device)
    R.section('Setup')
    R.kv([('Checkpoint', f'{args.model_path}' + (f'  (epoch {ck["epoch"]}, val loss {ck["val_loss"]:.4f})'
                                                   if 'epoch' in ck and 'val_loss' in ck else '')),
          ('Data', f'{args.data_dir}  (test: {len(test_dl.dataset)} images)'),
          ('Device', env.get('gpu', env['cpu']) + f' ({device})'),
          ('Output', out)])

    rep = Report(out)
    metrics = {'checkpoint': {'path': args.model_path, **ck}, 'environment': env, 'img_size': img_size}

    test = collect_predictions(model, test_dl, device, img_size, thresholds)
    save_predictions(test, os.path.join(out, 'predictions', 'test.csv'))
    m_test = all_metrics(test, thresholds)
    metrics['test'] = m_test

    rep.heading('1. Scene classification and gates (test)')
    report_condition(rep, m_test, 'test')

    loc = m_test['localisation']
    rep.heading('2. Hand localisation by scene (test)')
    rep.table('test_hand_iou_by_scene', *table_iou(loc['hand_iou']))
    rep.heading('3. Target localisation (test, pointing at target)')
    rep.note('Gap = distance from the fingertip to the target box (on-object: touching, near: <= 0.08 of '
             'the image, far: beyond).')
    rep.table('test_target_iou', *table_iou(loc['target_iou']))
    rep.heading('4. Fingertip and pointing-ray error by scene (test)')
    rep.table('test_fingertip_error', *table_err(loc['fingertip_px'], 'px', (5, 10)))
    rep.note(f'Pixels at the {img_size}x{img_size} model input. Ray error for "Pointing, no target" is measured '
             f'against the hand-centre-to-fingertip direction.')
    rep.table('test_ray_error', *table_err(loc['ray_deg'], 'deg', (15, 30)))
    gap_rows = [[b, d['n'], f'{d["fingertip_px"].get("mean", 0):.2f}' if d['n'] else '-',
                 f'{d["ray_deg"].get("mean", 0):.1f}' if d['n'] else '-', d['missed_pointing']]
                for b, d in loc['target_by_gap_detail'].items()]
    rep.table('test_gap_detail', ['Gap', 'n', 'Fingertip mean px', 'Ray mean deg', 'Missed pointing'], gap_rows)

    compare = {'Test (default thresholds)': m_test}
    section = 5
    if args.calibrate_thresholds:
        val_dl = get_eval_loader(args.data_dir, 'val', img_size, args.batch_size)
        val = collect_predictions(model, val_dl, device, img_size, thresholds)
        save_predictions(val, os.path.join(out, 'predictions', 'val.csv'))
        cal = calibrate_thresholds(val)
        m_cal = all_metrics(rescore(test, cal), cal)
        metrics['val'] = all_metrics(val, thresholds)
        metrics['test_calibrated'] = m_cal
        compare['Test (val-calibrated thresholds)'] = m_cal
        rep.heading(f'{section}. Val-calibrated gate thresholds (test)')
        rep.note(f'Highest threshold per gate with no missed positives on the {len(val)} validation images: '
                 f'hand {cal[0]:.2f}, pointing {cal[1]:.2f}, target {cal[2]:.2f}.')
        report_condition(rep, m_cal, 'test_calibrated')
        section += 1

    if args.multihand:
        mh_dl = get_multihand_loader(args.data_dir, 'test', img_size, args.batch_size, 0, args.seed)
        mh = collect_predictions(model, mh_dl, device, img_size, thresholds)
        save_predictions(mh, os.path.join(out, 'predictions', 'test_multihand.csv'))
        m_mh = all_metrics(mh, thresholds, with_ownership=True)
        metrics['test_multihand'] = m_mh
        compare['Test + distractor hands'] = m_mh
        rep.heading(f'{section}. Robustness to other people\'s hands (test with 1-2 pasted distractor hands)')
        report_condition(rep, m_mh, 'test_multihand')
        rep.table('test_multihand_hand_iou_by_scene', *table_iou(m_mh['localisation']['hand_iou']))
        o = m_mh['ownership']
        own_rows = [['Hand images with distractors', o['hand_images_with_distractors']],
                    ['Ownership accuracy % (predicted hand nearer the wearer)', R.pct(o['ownership_accuracy'], 2)],
                    ['Ownership accuracy 95% CI', '-' if o['ownership_accuracy_ci'][0] is None else
                     f'{R.pct(o["ownership_accuracy_ci"][0])}-{R.pct(o["ownership_accuracy_ci"][1])}'],
                    ['Predicted hand centre inside a distractor %', R.pct(o['hand_centre_on_distractor_rate'], 2)],
                    ['Pointing images with distractors', o['pointing_images_with_distractors']],
                    ['Ownership accuracy on pointing images %', R.pct(o['pointing_ownership_accuracy'], 2)],
                    ['No-hand images with distractors', o['no_hand_images_with_distractors']],
                    ['Still classified "No hand" %', R.pct(o['no_hand_still_no_hand_rate'], 2)]]
        rep.table('test_multihand_ownership', ['Ownership', 'Value'], own_rows)
        section += 1

    if len(compare) > 1:
        rep.heading(f'{section}. Headline comparison')
        rep.table('comparison', *table_compare(compare))
        section += 1

    rep.heading(f'{section}. Efficiency')
    profile = R.model_profile(model, img_size)
    sizes = R.serialized_sizes_kb(model)
    lat = {'CPU, 1 thread': R.measure_latency(model, img_size, 'cpu', threads=1),
           f'CPU, {os.cpu_count()} threads': R.measure_latency(model, img_size, 'cpu')}
    if device.type == 'cuda':
        lat[f'GPU ({env.get("gpu", "cuda")})'] = R.measure_latency(model, img_size, 'cuda')
    metrics['efficiency'] = {**profile, 'size_kb': sizes, 'latency': lat}
    eff_rows = [['Parameters', f'{profile["params_total"]:,}'],
                ['  backbone / heads', f'{profile["params_backbone"]:,} / {profile["params_heads"]:,}'],
                ['Compute per image', '-' if profile['gflops'] is None else f'{profile["gflops"]:.3f} GFLOPs'],
                ['Input', f'{img_size} x {img_size} RGB'],
                ['Size FP32 / FP16 / INT8', ' / '.join(f'{sizes[k]:.0f} KB' for k in ('fp32', 'fp16', 'int8'))]]
    eff_rows += [[f'Latency {k}, batch 1', f'{v["median_ms"]:.2f} ms median, {v["p90_ms"]:.2f} ms p90 '
                                           f'({v["fps"]:.0f} FPS)'] for k, v in lat.items()]
    rep.table('efficiency', ['Property', 'Value'], eff_rows)
    rep.note(f'CPU: {env["cpu"]}. Latency is PyTorch eager mode, not an on-device runtime.')

    # figures
    fdir = os.path.join(out, 'figures')
    fig_confusion(m_test, os.path.join(fdir, 'confusion_matrix'))
    fig_gate_curves(test, m_test, os.path.join(fdir, 'gate_roc'), os.path.join(fdir, 'gate_pr'))
    fig_iou_success(test, os.path.join(fdir, 'iou_success_rate'))
    fig_error_cdf(test, 'tip_err_px', f'Fingertip error (px at {img_size}x{img_size})', 'Fingertip error CDF',
                  os.path.join(fdir, 'fingertip_error_cdf'))
    fig_error_cdf(test, 'ray_err_deg', 'Ray angle error (deg)', 'Pointing-ray error CDF',
                  os.path.join(fdir, 'ray_error_cdf'))
    fig_target_by_gap(m_test, os.path.join(fdir, 'target_iou_by_gap'))
    if 'test_calibrated' in metrics:
        fig_confusion(metrics['test_calibrated'], os.path.join(fdir, 'confusion_matrix_calibrated'))
    if 'test_multihand' in metrics:
        fig_confusion(metrics['test_multihand'], os.path.join(fdir, 'confusion_matrix_multihand'))
    if args.num_vis > 0:
        fig_qualitative(test, test_dl.dataset, img_size, os.path.join(fdir, 'qualitative_examples'),
                        per_scene=max(1, args.num_vis // 4), seed=args.seed)

    metrics['runtime_s'] = time.time() - t_start
    R.write_json(os.path.join(out, 'metrics.json'), metrics)
    rep.save()
    R.section('Saved')
    R.kv([('metrics.json', 'every number above, plus validation and CI details'),
          ('summary.md', 'all tables in Markdown'),
          ('tables/', 'each table as CSV'),
          ('predictions/', 'per-image predictions and errors (recompute any metric without the model)'),
          ('figures/', 'PDF (for the paper) and PNG')])
    print(f'\n  Done in {R.fmt_duration(metrics["runtime_s"])}. Results in {out}\n')
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate SpatialNet',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model_path', type=str, default=os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
    p.add_argument('--data_dir', type=str, default=cfg.DATA_DIR)
    p.add_argument('--output_dir', type=str, default=os.path.join(cfg.OUTPUT_DIR, 'evaluation'))
    p.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    p.add_argument('--num_vis', type=int, default=16, help='qualitative examples (0 disables)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hc_thresh', type=float, default=0.5)
    p.add_argument('--pc_thresh', type=float, default=0.5)
    p.add_argument('--rc_thresh', type=float, default=0.5)
    p.add_argument('--calibrate_thresholds', action='store_true')
    p.add_argument('--multihand', action='store_true',
                   help='also evaluate the test split with deterministic pasted distractor hands')
    add_model_args(p)
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
