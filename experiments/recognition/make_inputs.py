"""Build the input files of the downstream recognition test (R1) from the saved per-image predictions.

    python experiments/recognition/make_inputs.py <out_dir>

Writes to <out_dir>, which is uploaded as the private Kaggle dataset spatialnet-recognition-inputs:
  boxes.csv       one row per test image with a target: labelled category, original image size, gap bin,
                  labelled box and every model's predicted target box (normalised cx, cy, w, h; empty if none)
  categories.txt  the label vocabulary: every non-empty target category in EgoPoint (all splits)
Boxes are clipped exactly as in revised_paper/scripts/detector_controls.py, so the crops show the boxes
that the paper's IoU metrics score.
"""
import csv
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from annotation import fingertip_roi_gap, parse_annotation  # noqa: E402

DATA = os.path.join(ROOT, 'EgoPoint_spatialnet')
SPATIALNET = {'spatialnet_s42': 'runs/main_run', 'spatialnet_s1': 'runs/seed_1', 'spatialnet_s2': 'runs/seed_2'}
YOLO = {'yolo_imagenet': 'runs/Y1_yolo_imagenet', 'yolo_imagenet_paste': 'runs/Y2_yolo_imagenet_paste',
        'yolo_coco': 'runs/Y3_yolo_coco', 'yolo_imagenet_s1': 'runs/Y4_yolo_imagenet_seed1'}


def clip(b):
    return None if b is None else [min(1, max(0, b[0])), min(1, max(0, b[1])),
                                   min(1, max(1e-3, b[2])), min(1, max(1e-3, b[3]))]


def box(r, prefix):
    try:
        return clip([float(r[f'{prefix}_{k}']) for k in ('cx', 'cy', 'w', 'h')])
    except (KeyError, ValueError, TypeError):
        return None


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return {r['name']: r for r in csv.DictReader(f)}


def main(out):
    os.makedirs(out, exist_ok=True)
    manifest = list(csv.DictReader(open(os.path.join(DATA, 'manifest.csv'), encoding='utf-8')))
    meta = {r['name']: r for r in manifest if r['split'] == 'test'}
    vocab = sorted({r['target_category'].strip() for r in manifest if r['T'] == '1' and r['target_category'].strip()})
    preds = {}
    for k, run in SPATIALNET.items():
        rows = read_csv(glob.glob(os.path.join(ROOT, run, '*', 'evaluation', 'predictions', 'test.csv'))[0])
        preds[k] = {n: box(r, 'pred_target') for n, r in rows.items()}
    for k, run in YOLO.items():
        rows = read_csv(glob.glob(os.path.join(ROOT, run, '**', 'predictions_test.csv'), recursive=True)[0])
        preds[k] = {n: box(r, 'target') for n, r in rows.items()}
    sources = list(preds)
    cols = ['name', 'category', 'orig_w', 'orig_h', 'bin', 'gt_area'] + \
        [f'{s}_{k}' for s in ['gt'] + sources for k in ('cx', 'cy', 'w', 'h')]
    n, skipped = 0, 0
    with open(os.path.join(out, 'boxes.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for lf in sorted(os.listdir(os.path.join(DATA, 'test', 'labels'))):
            name = lf[:-4]
            a = parse_annotation(os.path.join(DATA, 'test', 'labels', lf))
            roi = clip(a['roi'])
            if roi is None:
                continue
            cat = meta[name]['target_category'].strip()
            if not cat:
                skipped += 1
                continue
            gap = fingertip_roi_gap(a['fingertip'], roi)
            b = 'overlap' if gap == 0 else 'near' if 0 < gap <= 0.08 else 'large'
            row = [name, cat, meta[name]['orig_width'], meta[name]['orig_height'], b, roi[2] * roi[3]] + roi
            for s in sources:
                p = preds[s].get(name)
                row += p if p is not None else ['', '', '', '']
            w.writerow(row)
            n += 1
    with open(os.path.join(out, 'categories.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab) + '\n')
    with open(os.path.join(out, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'title': 'spatialnet-recognition-inputs', 'id': 'mosharafapurbo/spatialnet-recognition-inputs',
                   'licenses': [{'name': 'other'}]}, f, indent=2)
    print(f'{n} test images with a labelled category ({skipped} without); {len(vocab)} categories; '
          f'sources: {sources}')


if __name__ == '__main__':
    main(sys.argv[1])
