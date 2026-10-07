"""
Read and write the per-image H P T annotation (format: docs/annotation_format.md).
Imported by dataset.py, reporting.py, the Kaggle notebook and egopoint_preprocessing/.
"""

import os
import math

VALID_FLAGS = {(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)}


def fingertip_roi_gap(kp, roi):
    """Normalised distance from the fingertip to the target box (0 if inside); -1 if either is missing."""
    if kp is None or roi is None:
        return -1.0
    x1, y1, x2, y2 = roi[0] - roi[2] / 2, roi[1] - roi[3] / 2, roi[0] + roi[2] / 2, roi[1] + roi[3] / 2
    return float(math.hypot(max(x1 - kp[0], 0.0, kp[0] - x2), max(y1 - kp[1], 0.0, kp[1] - y2)))


def gap_bin(gap):
    """Fingertip-target gap stratum used in the dataset statistics and the evaluation."""
    if gap < 0:
        return 'n/a'
    if gap == 0.0:
        return 'on-object'
    if gap <= 0.08:
        return 'near'
    return 'far'


def parse_annotation(path):
    """-> {'hand', 'fingertip', 'roi', 'flags'}; absent parts are None, flags = (H, P, T).
    Raises ValueError naming the file if it breaks the format rules."""
    with open(path, 'r') as f:
        tok = f.read().split()
    if len(tok) != 13:
        raise ValueError(f'{path}: expected 13 numbers, found {len(tok)}')
    try:
        flags = tuple(int(v) for v in tok[:3])
        v = [float(x) for x in tok[3:]]
    except ValueError:
        raise ValueError(f'{path}: non-numeric value')
    if flags not in VALID_FLAGS:
        raise ValueError(f'{path}: invalid flags H P T = {flags}')
    h, p, t = flags
    hand, tip, roi = v[0:4], v[4:6], v[6:10]
    for name, on, vals, box in (('hand', h, hand, True), ('fingertip', p, tip, False),
                                ('roi', t, roi, True)):
        if not on:
            continue
        if any(x < 0 or x > 1 for x in vals):
            raise ValueError(f'{path}: {name} values outside [0, 1]')
        if box and (vals[2] <= 0 or vals[3] <= 0):
            raise ValueError(f'{path}: {name} box has zero size')
    return {'hand': hand if h else None, 'fingertip': tip if p else None,
            'roi': roi if t else None, 'flags': flags}


def copy_subset(data_dir, out_dir, per_split, seed=0):
    """Copy a small, scene-balanced subset (per_split = {'train': n, ...} images) for a quick pipeline check."""
    import random
    import shutil
    rng = random.Random(seed)
    for split, n in per_split.items():
        ldir, idir = os.path.join(data_dir, split, 'labels'), os.path.join(data_dir, split, 'images')
        images = {os.path.splitext(f)[0]: f for f in os.listdir(idir)}
        groups = {}
        for lf in sorted(os.listdir(ldir)):
            if lf.endswith('.txt') and lf[:-4] in images:
                groups.setdefault(parse_annotation(os.path.join(ldir, lf))['flags'], []).append(lf[:-4])
        take = []
        for names in groups.values():
            take += rng.sample(names, min(len(names), max(1, n // len(groups))))
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)
        for name in take:
            shutil.copy(os.path.join(ldir, name + '.txt'), os.path.join(out_dir, split, 'labels'))
            shutil.copy(os.path.join(idir, images[name]), os.path.join(out_dir, split, 'images'))


def _num(x):
    return '0' if x == 0 else repr(float(x))


def format_annotation(hand=None, fingertip=None, roi=None):
    """Annotation text; flags are set by which parts are given."""
    flags = (int(hand is not None), int(fingertip is not None), int(roi is not None))
    if flags not in VALID_FLAGS:
        raise ValueError(f'invalid combination H P T = {flags}')
    lines = [' '.join(map(str, flags)),
             ' '.join(_num(x) for x in (hand or [0, 0, 0, 0])),
             ' '.join(_num(x) for x in (fingertip or [0, 0])),
             ' '.join(_num(x) for x in (roi or [0, 0, 0, 0]))]
    return '\n'.join(lines) + '\n'
