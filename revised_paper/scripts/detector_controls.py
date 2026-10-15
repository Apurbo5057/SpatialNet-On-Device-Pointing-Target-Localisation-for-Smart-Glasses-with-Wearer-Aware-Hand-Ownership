"""Score SpatialNet and the controlled YOLO26n runs from per-image outputs, with paired-bootstrap intervals.

    python revised_paper/scripts/detector_controls.py runs/main_run runs/Y1_yolo_imagenet \
        runs/Y2_yolo_imagenet_paste runs/Y3_yolo_coco

Every model is scored by the same rules on the same images:
  - target metrics on the test images with a target (clean, and with the seed-42 pasted hands);
  - hand-gate threshold per model: the value that maximises no-hand vs hand accuracy on the clean val split;
  - per-source P@0.5 on COCO-derived and other test images (docs/egopoint_sources.csv).
Paired 95% intervals: 1,000 bootstrap resamples over images, seed 0. Writes runs/detector_controls.json,
or the file given by --out=<path>. The second YOLO26n seed (Y4) is scored in separate calls, so that
the intervals above stay unchanged:

    python revised_paper/scripts/detector_controls.py runs/main_run runs/Y4_yolo_imagenet_seed1 \
        --out=runs/detector_controls_y4.json
    python revised_paper/scripts/detector_controls.py runs/Y1_yolo_imagenet runs/Y4_yolo_imagenet_seed1 \
        --out=runs/detector_controls_y1_y4.json
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


def inside(p, b):
    return p is not None and b is not None and abs(p[0] - b[0]) <= b[2] / 2 and abs(p[1] - b[1]) <= b[3] / 2


def ground_truth(split):
    gt = {}
    for f in sorted(os.listdir(os.path.join(DATA, split, 'labels'))):
        a = parse_annotation(os.path.join(DATA, split, 'labels', f))
        hand, roi, tip = clip(a['hand']), clip(a['roi']), a['fingertip']
        gap = fingertip_roi_gap(tip, roi) if roi is not None and tip is not None else -1
        gt[f[:-4]] = {'hand': hand, 'roi': roi,
                      'bin': 'overlap' if gap == 0 else 'near' if 0 < gap <= 0.08 else 'large' if gap > 0.08 else None}
    return gt


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return {r['name']: r for r in csv.DictReader(f)}


def box(r, prefix):
    try:
        return [float(r[f'{prefix}_{k}']) for k in ('cx', 'cy', 'w', 'h')]
    except (KeyError, ValueError, TypeError):
        return None


def load_spatialnet(run):
    ev = glob.glob(os.path.join(run, '*', 'evaluation', 'predictions'))[0]
    out = {}
    for tag, fn in (('val', 'val.csv'), ('test', 'test.csv'), ('test_pasted', 'test_multihand.csv')):
        rows = read_csv(os.path.join(ev, fn))
        out[tag] = {n: {'hand_score': float(r['hand_prob']), 'hand': box(r, 'pred_hand'),
                        'target': box(r, 'pred_target')} for n, r in rows.items()}
    return out


def load_yolo(run):
    d = glob.glob(os.path.join(run, '**', 'predictions_test.csv'), recursive=True)[0]
    d = os.path.dirname(d)
    out = {}
    for tag in ('val', 'test', 'test_pasted'):
        rows = read_csv(os.path.join(d, f'predictions_{tag}.csv'))
        out[tag] = {n: {'hand_score': float(r['hand_conf']), 'hand': box(r, 'hand'),
                        'target': box(r, 'target')} for n, r in rows.items()}
    dist = read_csv(os.path.join(d, 'test_pasted_distractors.csv'))
    out['distractors'] = {n: json.loads(r['distractors']) for n, r in dist.items()}
    info = os.path.join(d, 'info.json')
    out['info'] = json.load(open(info)) if os.path.exists(info) else {}
    return out


def tune_threshold(pred, gt):
    names = [n for n in pred if n in gt]
    s = np.array([pred[n]['hand_score'] for n in names])
    y = np.array([gt[n]['hand'] is not None for n in names])
    grid = np.unique(np.concatenate([[0.5], s]))
    acc = [((s >= t) == y).mean() for t in grid]
    return float(grid[int(np.argmax(acc))])


def per_image(model, gt_test, sources, distractors):
    """Per-image indicator arrays keyed by metric, over the images each metric is defined on."""
    test, pasted = model['test'], model['test_pasted']
    pos = [n for n in gt_test if gt_test[n]['roi'] is not None and n in test and n in pasted]
    neg = [n for n in gt_test if gt_test[n]['hand'] is None and n in pasted]
    t = model['threshold']
    v = {}
    ti = {n: iou(test[n]['target'], gt_test[n]['roi']) for n in pos}
    tp = {n: iou(pasted[n]['target'], gt_test[n]['roi']) for n in pos}
    v['P@0.5'] = {n: 100.0 * (ti[n] >= .5) for n in pos}
    v['P@0.7'] = {n: 100.0 * (ti[n] >= .7) for n in pos}
    v['mIoU'] = {n: 100.0 * ti[n] for n in pos}
    v['Large gap'] = {n: 100.0 * (ti[n] >= .5) for n in pos if gt_test[n]['bin'] == 'large'}
    v['Hand mIoU'] = {n: 100.0 * iou(test[n]['hand'], gt_test[n]['hand']) for n in pos}
    v['Centre in'] = {n: 100.0 * inside((test[n]['target'] or [None])[:2] if test[n]['target'] else None,
                                        gt_test[n]['roi']) for n in pos}
    v['P@0.5 pasted'] = {n: 100.0 * (tp[n] >= .5) for n in pos}
    v['Pasted change'] = {n: v['P@0.5 pasted'][n] - v['P@0.5'][n] for n in pos}
    v['No hand kept, clean'] = {n: 100.0 * (test[n]['hand_score'] < t) for n in neg if n in test}
    v['No hand kept, pasted'] = {n: 100.0 * (pasted[n]['hand_score'] < t) for n in neg}
    own = {}
    for n in pos:
        ds = distractors.get(n) or []
        hb = pasted[n]['hand']
        if ds and hb is not None:
            d_w = np.hypot(hb[0] - gt_test[n]['hand'][0], hb[1] - gt_test[n]['hand'][1])
            d_o = min(np.hypot(hb[0] - d[0], hb[1] - d[1]) for d in ds)
            own[n] = 100.0 * (d_w < d_o)
    v['Ownership, pasted'] = own
    coco = {n for n, s in sources.items() if s}
    v['P@0.5, COCO-derived'] = {n: v['P@0.5'][n] for n in pos if n in coco}
    v['P@0.5, other'] = {n: v['P@0.5'][n] for n in pos if n not in coco}
    return v


def mean(d):
    return float(np.mean(list(d.values()))) if d else float('nan')


def paired(a, b, rng):
    """mean(b) - mean(a) over shared images, with a bootstrap 95% interval."""
    idx = sorted(set(a) & set(b))
    x, y = np.array([a[i] for i in idx]), np.array([b[i] for i in idx])
    d = y - x
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(idx)


def did(a, b, group1, group2, rng):
    """[(b - a) on group1] - [(b - a) on group2]: each group resampled separately."""
    d1 = np.array([b[i] - a[i] for i in sorted(group1) if i in a and i in b])
    d2 = np.array([b[i] - a[i] for i in sorted(group2) if i in a and i in b])
    bs = [d1[rng.integers(0, len(d1), len(d1))].mean() - d2[rng.integers(0, len(d2), len(d2))].mean()
          for _ in range(B)]
    return float(d1.mean() - d2.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main(runs, out):
    gt_val, gt_test = ground_truth('val'), ground_truth('test')
    with open(os.path.join(ROOT, 'docs', 'egopoint_sources.csv'), encoding='utf-8') as f:
        sources = {r['name']: r['coco_source'] for r in csv.DictReader(f) if r['split'] == 'test'}
    models, dist = {}, {}
    for run in runs:
        name = os.path.basename(run.rstrip('/\\'))
        m = load_yolo(run) if glob.glob(os.path.join(run, '**', 'predictions_test.csv'), recursive=True) \
            else load_spatialnet(run)
        m['threshold'] = tune_threshold(m['val'], gt_val)
        models[name] = m
        if 'distractors' in m:
            dist = m['distractors']
    res = {'thresholds': {k: m['threshold'] for k, m in models.items()},
           'info': {k: m.get('info', {}) for k, m in models.items()}, 'metrics': {}, 'paired': {}}
    vals = {k: per_image(m, gt_test, sources, dist) for k, m in models.items()}
    for k, v in vals.items():
        res['metrics'][k] = {m: round(mean(d), 2) for m, d in v.items()}
        res['metrics'][k]['n_pos'] = len(v['P@0.5'])
    names = list(models)
    ref = names[0]
    rng = np.random.default_rng(SEED)
    for k in names[1:]:
        res['paired'][f'{k} - {ref}'] = {
            m: dict(zip(('diff', 'ci', 'n'), paired(vals[ref][m], vals[k][m], rng)))
            for m in ('P@0.5', 'P@0.7', 'mIoU', 'Large gap', 'Hand mIoU', 'Centre in', 'Pasted change',
                      'No hand kept, pasted', 'P@0.5, COCO-derived', 'P@0.5, other')}
    by = {n.split('_', 1)[0]: n for n in names}
    if 'Y1' in by and 'Y2' in by:
        res['paired']['pasting effect: Y2 - Y1'] = {
            m: dict(zip(('diff', 'ci', 'n'), paired(vals[by['Y1']][m], vals[by['Y2']][m], rng)))
            for m in ('P@0.5', 'P@0.7', 'Pasted change', 'P@0.5 pasted', 'No hand kept, pasted',
                      'Ownership, pasted')}
    if 'Y1' in by and 'Y3' in by:
        a, b = vals[by['Y1']]['P@0.5'], vals[by['Y3']]['P@0.5']
        coco = set(vals[by['Y1']]['P@0.5, COCO-derived'])
        other = set(vals[by['Y1']]['P@0.5, other'])
        d_c = paired({i: a[i] for i in coco}, {i: b[i] for i in coco}, rng)
        d_o = paired({i: a[i] for i in other}, {i: b[i] for i in other}, rng)
        dd, ci = did(a, b, coco, other, rng)
        res['paired']['COCO pretraining: Y3 - Y1'] = {
            'P@0.5, COCO-derived': dict(zip(('diff', 'ci', 'n'), d_c)),
            'P@0.5, other': dict(zip(('diff', 'ci', 'n'), d_o)),
            'difference (COCO-derived minus other)': {'diff': dd, 'ci': ci}}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    cols = ['P@0.5', 'P@0.7', 'mIoU', 'Large gap', 'Hand mIoU', 'Centre in', 'Pasted change',
            'No hand kept, pasted', 'Ownership, pasted', 'P@0.5, COCO-derived', 'P@0.5, other']
    print('| Model | ' + ' | '.join(cols) + ' |')
    for k in names:
        print(f'| {k} | ' + ' | '.join(f'{res["metrics"][k][c]:.1f}' for c in cols) + ' |')
    print(json.dumps({'thresholds': res['thresholds'], 'paired': res['paired']}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    args = sys.argv[1:]
    out = next((a[len('--out='):] for a in args if a.startswith('--out=')),
               os.path.join(ROOT, 'runs', 'detector_controls.json'))
    main([a for a in args if not a.startswith('--out=')], out)
