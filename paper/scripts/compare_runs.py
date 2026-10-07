"""Compare experiment runs with the main run on the same test images (paired bootstrap).

    python paper/scripts/compare_runs.py runs/main_run runs/img224 runs/A1_no_copypaste

The first run is the reference. For each other run: the metric and its difference from the reference,
with a 95% paired-bootstrap interval (1,000 resamples over images). Fingertip error is given in % of the
image side, so runs with different input sizes are comparable. Writes <run>/comparison_vs_<ref>.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

B, SEED = 1000, 0


def load(run):
    out = glob.glob(os.path.join(run, '*', 'evaluation'))[0]
    m = json.load(open(os.path.join(out, 'metrics.json'), encoding='utf-8'))
    size = m.get('img_size', 160)
    t = pd.read_csv(os.path.join(out, 'predictions', 'test.csv')).set_index('name')
    mh = pd.read_csv(os.path.join(out, 'predictions', 'test_multihand.csv')).set_index('name')
    t['tip_pct'] = 100 * t['tip_err_px'] / size
    return {'name': os.path.basename(run.rstrip('/\\')), 'size': size, 'test': t, 'mh': mh}


def per_image(r):
    """Per-image arrays; each metric is a mean over its own image subset."""
    t, mh = r['test'], r['mh']
    pos = t[t.gt_T == 1]
    neg = t[t.gt_H == 0]
    far = pos[pos.gap_bin == 'far']
    near = pos[pos.gap_bin == 'near']
    on = pos[pos.gap_bin == 'on-object']
    mh_hand = mh[(mh.gt_H == 1) & (mh.n_distractors > 0)]
    mh_neg = mh[mh.gt_H == 0]
    return {
        'Target mIoU': 100 * pos.target_iou, 'Target P@0.3': 100 * (pos.target_iou >= .3),
        'Target P@0.5': 100 * (pos.target_iou >= .5), 'Target P@0.7': 100 * (pos.target_iou >= .7),
        'On-object P@0.5': 100 * (on.target_iou >= .5), 'Near P@0.5': 100 * (near.target_iou >= .5),
        'Far P@0.5': 100 * (far.target_iou >= .5), 'Far mIoU': 100 * far.target_iou,
        'Hand mIoU': 100 * pos.hand_iou, 'Hand P@0.5': 100 * (pos.hand_iou >= .5),
        'Fingertip error (% of side)': pos.tip_pct, 'Direction error (deg)': pos.ray_err_deg,
        'Direction error, far (deg)': far.ray_err_deg,
        'Hand gate: pointing frames missed (%)': 100 * (pos.pred_scene == 'No hand'),
        'Hand gate: no-hand false alarms (%)': 100 * (neg.pred_scene != 'No hand'),
        'Ownership accuracy, pasted hands (%)': 100 * mh_hand.owner_correct,
        'Centre on pasted hand (%)': 100 * mh_hand.pred_on_distractor,
        'No-hand kept with pasted hands (%)': 100 * (mh_neg.pred_scene == 'No hand'),
        'Target P@0.5 with pasted hands': 100 * (mh[mh.gt_T == 1].target_iou >= .5),
    }


def paired(a, b, rng):
    """Mean of b, mean of a, and a bootstrap 95% CI of mean(b) - mean(a) over the shared images."""
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx].astype(float).values, b.loc[idx].astype(float).values
    d = y - x
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return len(idx), x.mean(), y.mean(), d.mean(), lo, hi


def main(runs):
    ref = load(runs[0])
    ra = per_image(ref)
    for path in runs[1:]:
        r = load(path)
        rb = per_image(r)
        rng = np.random.default_rng(SEED)
        rows = []
        print(f'\n== {r["name"]} ({r["size"]} px)  vs  {ref["name"]} ({ref["size"]} px)')
        print(f'{"Metric":42s}{"n":>6s}{ref["name"][:10]:>11s}{r["name"][:10]:>11s}{"diff":>8s}  95% CI')
        for k in ra:
            n, x, y, d, lo, hi = paired(ra[k], rb[k], rng)
            sig = '*' if lo > 0 or hi < 0 else ' '
            print(f'{k:42s}{n:6d}{x:11.2f}{y:11.2f}{d:+8.2f}  [{lo:+.2f}, {hi:+.2f}] {sig}')
            rows.append({'metric': k, 'n': n, 'reference': x, 'run': y, 'diff': d, 'ci_low': lo, 'ci_high': hi,
                         'significant': bool(lo > 0 or hi < 0)})
        out = os.path.join(path, f'comparison_vs_{ref["name"]}.json')
        json.dump({'reference': ref['name'], 'run': r['name'], 'img_size': r['size'], 'rows': rows}, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
