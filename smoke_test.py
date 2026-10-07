"""
End-to-end smoke test for the SpatialNet pipeline: builds a small synthetic
dataset and runs train, evaluate, ONNX export and inference via the CLIs.

    python smoke_test.py [--keep] [--epochs 2] [--img_size 160]
"""

import os
import sys
import math
import random
import shutil
import argparse
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from annotation import format_annotation

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SIZE = 224


def _draw_hand(d, cx, cy, w, h, rng, from_bottom=True):
    """Crude forearm + hand + extended index finger. Returns the fingertip."""
    x1, y1 = (cx - w / 2) * SIZE, (cy - h / 2) * SIZE
    x2, y2 = (cx + w / 2) * SIZE, (cy + h / 2) * SIZE
    skin = (215 + rng.randint(-30, 25), 165 + rng.randint(-30, 25), 130 + rng.randint(-25, 25))
    d.ellipse([x1, y1 + (y2 - y1) * 0.35, x2, y2], fill=skin)
    if from_bottom:
        d.polygon([(x1 + (x2 - x1) * 0.25, y2), (x2 - (x2 - x1) * 0.25, y2),
                   (x2 - (x2 - x1) * 0.15, SIZE), (x1 + (x2 - x1) * 0.15, SIZE)], fill=skin)
        tipx, tipy = (x1 + x2) / 2 + rng.uniform(-6, 6), y1
    else:
        tipx, tipy = x2, (y1 + y2) / 2
    d.ellipse([tipx - 5, tipy - 5, tipx + 5, tipy + 5], fill=skin)
    return tipx / SIZE, tipy / SIZE


def _make_sample(rng, flags, with_far_target):
    """flags = (H, P, T) -> (PIL image, hand box, fingertip, target box); absent parts are None."""
    has_hand, pointing, has_target = flags
    img = Image.new('RGB', (SIZE, SIZE), (200 + rng.randint(-40, 40),
                                          190 + rng.randint(-40, 40),
                                          180 + rng.randint(-40, 40)))
    d = ImageDraw.Draw(img)
    # distractor objects
    objs = []
    for _ in range(rng.randint(2, 4)):
        ow, oh = rng.uniform(0.10, 0.20), rng.uniform(0.10, 0.22)
        ocx, ocy = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.55)
        col = tuple(rng.randint(20, 235) for _ in range(3))
        d.rectangle([(ocx - ow / 2) * SIZE, (ocy - oh / 2) * SIZE,
                     (ocx + ow / 2) * SIZE, (ocy + oh / 2) * SIZE], fill=col)
        objs.append([ocx, ocy, ow, oh])

    if not has_hand:
        return img, None, None, None

    hw, hh = rng.uniform(0.22, 0.38), rng.uniform(0.30, 0.45)
    hcx = rng.uniform(0.25, 0.75)
    hcy = 1.0 - hh / 2 - rng.uniform(0.0, 0.08)     # wearer hand enters from the bottom
    tip = _draw_hand(d, hcx, hcy, hw, hh, rng)
    hand = [hcx, hcy, hw, hh]
    if not pointing:
        return img, hand, None, None

    kp = [min(max(v, 0.0), 1.0) for v in tip]
    if not has_target:
        return img, hand, kp, None

    if with_far_target:
        target = min(objs, key=lambda o: -math.hypot(o[0] - kp[0], o[1] - kp[1]))
    else:
        target = [kp[0], max(0.08, kp[1] - 0.06), rng.uniform(0.10, 0.16), rng.uniform(0.10, 0.16)]
        d.rectangle([(target[0] - target[2] / 2) * SIZE, (target[1] - target[3] / 2) * SIZE,
                     (target[0] + target[2] / 2) * SIZE, (target[1] + target[3] / 2) * SIZE],
                    outline=(255, 0, 0), width=3)
    return img, hand, kp, list(target)


def build_dataset(root, counts, seed=0):
    rng = random.Random(seed)
    for split, n in counts.items():
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(root, split, sub), exist_ok=True)
        for i in range(n):
            flags = [(1, 1, 1)] * 3 + [(1, 0, 0)] * 2 + [(0, 0, 0), (1, 1, 0)]
            flags = flags[i % 7]
            far = flags == (1, 1, 1) and (i % 6 == 0)
            img, hand, kp, roi = _make_sample(rng, flags, far)
            name = f'img_{split}_{i:04d}'
            img.save(os.path.join(root, split, 'images', name + '.jpg'), quality=92)
            with open(os.path.join(root, split, 'labels', name + '.txt'), 'w') as f:
                f.write(format_annotation(hand, kp, roi))
    print(f'  synthetic dataset at {root}: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))


def run(cmd, title):
    print(f'\n{"=" * 74}\n  {title}\n{"=" * 74}')
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    out = (r.stdout or '') + (r.stderr or '')
    print(out[-6000:])
    if r.returncode != 0:
        raise SystemExit(f'[FAIL] {title} exited {r.returncode}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--img_size', type=int, default=160)
    ap.add_argument('--epochs', type=int, default=2)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix='spatialnet_smoke_')
    data = os.path.join(tmp, 'dataset')
    out = os.path.join(tmp, 'outputs')
    try:
        print('=' * 74 + '\n  SpatialNet -- end-to-end smoke test\n' + '=' * 74)
        build_dataset(data, {'train': 84, 'val': 28, 'test': 28})

        run([PY, 'train.py', '--data_dir', data, '--output_dir', out,
             '--epochs', str(args.epochs), '--batch_size', '8', '--lr', '1e-3',
             '--img_size', str(args.img_size), '--num_workers', '0',
             '--patience', '99', '--quantize'], 'TRAIN')

        eval_out = run([PY, 'test_evaluate.py', '--model_path', os.path.join(out, 'best_model.pth'),
                        '--data_dir', data, '--output_dir', os.path.join(out, 'evaluation'),
                        '--img_size', str(args.img_size), '--batch_size', '8',
                        '--num_vis', '8', '--multihand', '--calibrate_thresholds'], 'EVALUATE')

        onnx_path = os.path.join(out, 'spatialnet.onnx')
        onnx_out = run([PY, 'export_onnx.py', '--model_path', os.path.join(out, 'best_model.pth'),
                        '--out', onnx_path, '--img_size', str(args.img_size)], 'ONNX EXPORT')

        infer_dir = os.path.join(tmp, 'infer')
        os.makedirs(infer_dir, exist_ok=True)
        src = os.path.join(data, 'test', 'images')
        for fn in sorted(os.listdir(src))[:3]:
            shutil.copy2(os.path.join(src, fn), os.path.join(infer_dir, fn))
        run([PY, 'inference.py', '--model_path', os.path.join(out, 'best_model.pth'),
             '--image_dir', infer_dir, '--img_size', str(args.img_size), '--heatmaps',
             '--out_dir', os.path.join(tmp, 'infer_results')], 'INFERENCE')

        ev = os.path.join(out, 'evaluation')
        tr = os.path.join(out, 'training')
        checks = {
            'multi-hand robustness reported': 'distractor hands' in eval_out,
            'ownership reported': 'Ownership accuracy' in eval_out,
            'gap strata reported': 'Gap far' in eval_out,
            'ONNX parity < 1e-3': _parity_ok(onnx_out),
            'checkpoints written': all(os.path.exists(os.path.join(out, f)) for f in
                                       ['best_model.pth', 'final_model.pth', 'model_fp16.pth']),
            'onnx file written': os.path.exists(onnx_path),
            'training stats written': all(os.path.exists(os.path.join(tr, f)) for f in
                                          ['history.csv', 'run_info.json', 'summary.md',
                                           'figures/training_curves.pdf']),
            'evaluation stats written': all(os.path.exists(os.path.join(ev, f)) for f in
                                            ['metrics.json', 'summary.md', 'predictions/test.csv',
                                             'tables/test_hand_iou_by_scene.csv',
                                             'figures/confusion_matrix.pdf']),
        }
        print('\n' + '=' * 74 + '\n  SMOKE TEST SUMMARY\n' + '=' * 74)
        for k, v in checks.items():
            print(f'  [{"OK  " if v else "FAIL"}] {k}')
        if not all(checks.values()):
            raise SystemExit('[FAIL] smoke test incomplete')
        size_mb = os.path.getsize(onnx_path) / 1024 ** 2
        print(f'\n  ONNX size: {size_mb:.2f} MB')
        print('  [OK] end-to-end pipeline healthy')
    finally:
        if args.keep:
            print(f'\n  artefacts kept at {tmp}')
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def _parity_ok(text):
    for line in text.splitlines():
        if 'max abs diff' in line:
            try:
                return float(line.strip().split()[-1]) < 1e-3
            except ValueError:
                return False
    return False


if __name__ == '__main__':
    main()
