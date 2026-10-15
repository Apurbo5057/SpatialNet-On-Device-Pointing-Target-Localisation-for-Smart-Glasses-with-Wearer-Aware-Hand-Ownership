"""'Simple references' rows of the main results table (paper Table 4), from the evaluation outputs.

  (a) the mean training target box at a fixed position;
  (b) a box of mean training size centred on SpatialNet's predicted fingertip;
  (c) the same box centred on the ground-truth fingertip (oracle).

Run from the repository root after test_evaluate.py:
  python paper/scripts/simple_references.py --pred <out>/evaluation/predictions/test.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from annotation import parse_annotation, fingertip_roi_gap, gap_bin  # noqa: E402


def iou(a, b):
    """IoU of two (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def labels(split, data_dir):
    d = os.path.join(data_dir, split, 'labels')
    return {f[:-4]: parse_annotation(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith('.txt')}


def summary(values, bins):
    v, b = np.asarray(values), np.asarray(bins)
    far = v[b == 'far']
    return (100 * v.mean(), 100 * (v >= 0.3).mean(), 100 * (v >= 0.5).mean(), 100 * (v >= 0.7).mean(),
            100 * (far >= 0.5).mean() if len(far) else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='evaluation/predictions/test.csv')
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'EgoPoint_spatialnet'))
    args = ap.parse_args()

    train = [a['roi'] for a in labels('train', args.data_dir).values() if a['roi'] is not None]
    mean_box = np.mean(np.asarray(train), axis=0)
    test = labels('test', args.data_dir)
    rows = [r for r in csv.DictReader(open(args.pred, encoding='utf-8')) if r['gt_T'] == '1']

    out = {'Mean training box, fixed position': [], 'Mean-size box at predicted fingertip': [],
           'Mean-size box at true fingertip (oracle)': [], 'SpatialNet (from predictions)': []}
    bins = []
    for r in rows:
        a = test[r['name']]
        gt = a['roi']
        bins.append(gap_bin(fingertip_roi_gap(a['fingertip'], gt)))
        px, py = float(r['pred_tip_x']), float(r['pred_tip_y'])
        out['Mean training box, fixed position'].append(iou(mean_box, gt))
        out['Mean-size box at predicted fingertip'].append(iou((px, py, mean_box[2], mean_box[3]), gt))
        out['Mean-size box at true fingertip (oracle)'].append(
            iou((a['fingertip'][0], a['fingertip'][1], mean_box[2], mean_box[3]), gt))
        pred = [float(r[k]) for k in ('pred_target_cx', 'pred_target_cy', 'pred_target_w', 'pred_target_h')]
        out['SpatialNet (from predictions)'].append(iou(pred, gt))

    print(f'{len(rows)} pointing-at-target test images; mean training box (cx, cy, w, h) = '
          + ', '.join(f'{v:.3f}' for v in mean_box))
    print(f"{'Reference':<44}{'IoU':>7}{'>=0.3':>7}{'>=0.5':>7}{'>=0.7':>7}{'Far>=0.5':>10}")
    for name, v in out.items():
        print(f'{name:<44}' + ''.join(f'{x:>7.1f}' for x in summary(v, bins)[:4]) + f'{summary(v, bins)[4]:>10.1f}')
    print('The last row should match test_evaluate.py; if it does not, check the predictions file.')


if __name__ == '__main__':
    main()
