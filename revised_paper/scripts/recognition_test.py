"""Score the downstream recognition test: is the labelled object named from each model's target crop?

    python revised_paper/scripts/recognition_test.py runs/R2_recognition

The run folder holds the Kaggle output of experiments/recognition/recognition_test.py
(recognition/per_crop_<recogniser>.csv). For every recogniser and crop protocol this prints top-1 and
top-5 accuracy per box source, paired differences between SpatialNet and each controlled detector
(1,000 bootstrap resamples over images, seed 0, as in detector_controls.py), the same by fingertip-target
gap, and how often a crop is recognised when the box is accurate (IoU >= 0.5), loose but centred on the
labelled object, or centred elsewhere. Writes <run>/recognition_test.json.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from annotation import fingertip_roi_gap, parse_annotation  # noqa: E402

DATA = os.path.join(ROOT, 'EgoPoint_spatialnet')
B, SEED = 1000, 0
SPATIALNET = 'spatialnet_s42'
DETECTORS = ['yolo_imagenet', 'yolo_imagenet_s1', 'yolo_imagenet_paste', 'yolo_coco']
PREDS = {'spatialnet_s42': ('runs/main_run', 'pred_target'), 'spatialnet_s1': ('runs/seed_1', 'pred_target'),
         'spatialnet_s2': ('runs/seed_2', 'pred_target'), 'yolo_imagenet': ('runs/Y1_yolo_imagenet', 'target'),
         'yolo_imagenet_paste': ('runs/Y2_yolo_imagenet_paste', 'target'), 'yolo_coco': ('runs/Y3_yolo_coco', 'target'),
         'yolo_imagenet_s1': ('runs/Y4_yolo_imagenet_seed1', 'target')}


def clip(b):
    return None if b is None else [min(1, max(0, b[0])), min(1, max(0, b[1])),
                                   min(1, max(1e-3, b[2])), min(1, max(1e-3, b[3]))]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-8)


def inside(b, g):
    return b is not None and abs(b[0] - g[0]) <= g[2] / 2 and abs(b[1] - g[1]) <= g[3] / 2


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def ground_truth():
    gt = {}
    for f in sorted(os.listdir(os.path.join(DATA, 'test', 'labels'))):
        a = parse_annotation(os.path.join(DATA, 'test', 'labels', f))
        roi = clip(a['roi'])
        if roi is None:
            continue
        gap = fingertip_roi_gap(a['fingertip'], roi)
        gt[f[:-4]] = {'roi': roi, 'bin': 'overlap' if gap == 0 else 'near' if 0 < gap <= 0.08 else 'large'}
    return gt


def predictions():
    out = {}
    for k, (run, prefix) in PREDS.items():
        path = (glob.glob(os.path.join(ROOT, run, '*', 'evaluation', 'predictions', 'test.csv')) or
                glob.glob(os.path.join(ROOT, run, '**', 'predictions_test.csv'), recursive=True))[0]
        rows = {r['name']: r for r in read_csv(path)}
        out[k] = {}
        for n, r in rows.items():
            try:
                out[k][n] = clip([float(r[f'{prefix}_{c}']) for c in ('cx', 'cy', 'w', 'h')])
            except (KeyError, ValueError, TypeError):
                out[k][n] = None
    return out


def paired(a, b, rng):
    """mean(b) - mean(a) over shared images, with a bootstrap 95% interval."""
    idx = sorted(set(a) & set(b))
    d = np.array([b[i] for i in idx]) - np.array([a[i] for i in idx])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return {'diff': round(float(d.mean()), 2), 'ci': [round(float(np.percentile(bs, 2.5)), 2),
                                                      round(float(np.percentile(bs, 97.5)), 2)], 'n': len(idx)}


def main(run):
    gt, preds = ground_truth(), predictions()
    files = sorted(glob.glob(os.path.join(run, '**', 'per_crop_*.csv'), recursive=True))
    assert files, f'no per_crop_*.csv under {run}'
    res = {'run': os.path.basename(run.rstrip('/\\')), 'recognisers': {}}
    rng = np.random.default_rng(SEED)
    for path in files:
        key = os.path.basename(path)[len('per_crop_'):-4]
        rows = read_csv(path)
        hit1, hit5 = {}, {}
        for r in rows:
            rank = int(r['rank'])
            hit1.setdefault((r['protocol'], r['source']), {})[r['name']] = 100.0 * (rank == 1)
            hit5.setdefault((r['protocol'], r['source']), {})[r['name']] = 100.0 * (1 <= rank <= 5)
        rec = {'accuracy': {}, 'paired': {}, 'by_gap': {}, 'by_box_quality': {}}
        for (p, s), d in sorted(hit1.items()):
            rec['accuracy'][f'{p}/{s}'] = {'top1': round(float(np.mean(list(d.values()))), 2),
                                           'top5': round(float(np.mean(list(hit5[(p, s)].values()))), 2),
                                           'n': len(d)}
        for p in sorted({p for p, _ in hit1 if p != 'whole'}):
            sn = hit1[(p, SPATIALNET)]
            for det in DETECTORS:
                if (p, det) not in hit1:
                    continue
                rec['paired'][f'{p}: {SPATIALNET} - {det}'] = {
                    'top1': paired(hit1[(p, det)], sn, rng), 'top5': paired(hit5[(p, det)], hit5[(p, SPATIALNET)], rng)}
                for g in ('overlap', 'near', 'large'):
                    names = {n for n in sn if gt[n]['bin'] == g}
                    rec['by_gap'][f'{p}: {SPATIALNET} - {det}, {g}'] = paired(
                        {n: hit1[(p, det)][n] for n in names}, {n: sn[n] for n in names}, rng)
            # How often a crop is recognised, by how well its box matches the labelled box.
            for s in [SPATIALNET] + DETECTORS:
                if (p, s) not in hit1:
                    continue
                groups = {'IoU>=0.5': [], 'IoU<0.5, centre in': [], 'centre out': []}
                for n, v in hit1[(p, s)].items():
                    b, g = preds[s].get(n), gt[n]['roi']
                    k = 'IoU>=0.5' if iou(b, g) >= 0.5 else 'IoU<0.5, centre in' if inside(b, g) else 'centre out'
                    groups[k].append(v)
                rec['by_box_quality'][f'{p}/{s}'] = {k: {'share': round(100 * len(v) / len(hit1[(p, s)]), 1),
                                                         'top1': round(float(np.mean(v)), 1) if v else None}
                                                     for k, v in groups.items()}
        res['recognisers'][key] = rec
        print(f'\n## {key}')
        print('| Protocol / source | Top-1 | Top-5 | n |')
        for k, v in rec['accuracy'].items():
            print(f'| {k} | {v["top1"]:.1f} | {v["top5"]:.1f} | {v["n"]} |')
        for k, v in rec['paired'].items():
            print(f'{k}: top-1 {v["top1"]["diff"]:+.1f} {v["top1"]["ci"]}, top-5 {v["top5"]["diff"]:+.1f} {v["top5"]["ci"]}')
        for k, v in rec['by_gap'].items():
            print(f'  {k}: {v["diff"]:+.1f} {v["ci"]} (n={v["n"]})')
        for k, v in rec['by_box_quality'].items():
            print(f'  {k}: ' + '; '.join(f'{g} {x["share"]}% of images, top-1 {x["top1"]}' for g, x in v.items()))
    out = os.path.join(run, 'recognition_test.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1])
