"""YOLO26n baselines for the SpatialNet paper, trained and scored under one recipe.

MODE
  imagenet        backbone initialised from ImageNet (yolo26n-cls weights); neck and head from scratch
  imagenet_paste  as imagenet, but trained with SpatialNet's distractor-hand pasting, applied on the fly
                  with the same code and probabilities (pasted hands stay unlabelled background)
  coco            COCO-pretrained yolo26n.pt

The recipe is shared by all modes: EgoPoint converted images, 2 classes (wearer hand, target), 128 px,
no mosaic/mixup/copy-paste, 150 epochs, seed 42. For val, test and the pasted test set (the same pasted
images as SpatialNet's multi-hand evaluation, seed 42), every image's top-scoring hand box and target box
are written with their confidences, so all metrics and paired comparisons are computed offline.
"""
MODE = 'imagenet'
QUICK = False
SEED = 42          # training seed; the pasted test set always uses seed 42

import csv
import json
import os
import shutil
import subprocess
import sys
import time

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'ultralytics'], check=True)

import math  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def find_input(*required):
    """First folder under /kaggle/input that contains every required path."""
    for root, dirs, _ in os.walk('/kaggle/input'):
        dirs[:] = [d for d in dirs if d not in ('images', 'labels')]
        if all(os.path.exists(os.path.join(root, r)) for r in required):
            return root
    raise FileNotFoundError(f'No attached input contains {required}')


CODE_SRC = find_input('dataset.py', 'annotation.py', 'config.py')
DATA_DIR = find_input('train/labels', 'val/labels', 'test/labels')
CODE_DIR = '/tmp/spatialnet_code'
shutil.rmtree(CODE_DIR, ignore_errors=True)
shutil.copytree(CODE_SRC, CODE_DIR)
sys.path.insert(0, CODE_DIR)
from annotation import parse_annotation  # noqa: E402
from dataset import KeypointPointingDataset  # noqa: E402

from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402

OUT = f'/kaggle/working/yolo_{MODE}'
YOLO_ROOT = '/tmp/yolo'
LIMIT = {'train': 96, 'val': 48, 'test': 64} if QUICK else None
EPOCHS = 1 if QUICK else 150
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
os.makedirs(OUT, exist_ok=True)
info = {'mode': MODE, 'quick': QUICK, 'seed': SEED, 'ultralytics': ULTRALYTICS_VERSION, 'torch': torch.__version__,
        'device': str(DEVICE), 'data_dir': DATA_DIR}
print(info, flush=True)


def split_names(split):
    names = sorted(f[:-4] for f in os.listdir(os.path.join(DATA_DIR, split, 'labels')) if f.endswith('.txt'))
    return names[:LIMIT[split]] if LIMIT else names


def image_path(split, name):
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = os.path.join(DATA_DIR, split, 'images', name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def clip(b):
    return KeypointPointingDataset._clip_bbox(b)


# 1. Dataset in YOLO format: class 0 = wearer's hand, class 1 = pointed-at target; no-hand images empty
shutil.rmtree(YOLO_ROOT, ignore_errors=True)
counts = {}
for split in ('train', 'val', 'test'):
    idir, ldir = f'{YOLO_ROOT}/images/{split}', f'{YOLO_ROOT}/labels/{split}'
    os.makedirs(idir)
    os.makedirs(ldir)
    names = split_names(split)
    for n in names:
        src = image_path(split, n)
        shutil.copy(src, os.path.join(idir, n + os.path.splitext(src)[1]))   # local disk: input mount is slow
        a = parse_annotation(os.path.join(DATA_DIR, split, 'labels', n + '.txt'))
        lines = [f'{c} ' + ' '.join(f'{v:.6f}' for v in clip(b))
                 for c, b in ((0, a['hand']), (1, a['roi'])) if b is not None]
        with open(os.path.join(ldir, n + '.txt'), 'w') as f:
            f.write('\n'.join(lines))
    counts[split] = len(names)
data_yaml = f'{YOLO_ROOT}/data.yaml'
with open(data_yaml, 'w') as f:
    f.write(f'path: {YOLO_ROOT}\ntrain: images/train\nval: images/val\ntest: images/test\n'
            'names:\n  0: hand\n  1: target\n')
info['images'] = counts
print('dataset', counts, flush=True)


# 2. On-the-fly distractor pasting with SpatialNet's own procedure (training images only)
class Paster:
    """Calls KeypointPointingDataset._maybe_copy_paste: p = 0.5 for hand images and 0.3 for no-hand images,
    1-2 pasted hands (30% self-paste), decoy patches with p = 0.15; labels never change."""

    def __init__(self):
        self.ds = KeypointPointingDataset(DATA_DIR, 'train', img_size=160, augment=True, copy_paste=0.5)
        self.index = {s['name']: i for i, s in enumerate(self.ds.samples)}

    def __call__(self, path, im_bgr):
        idx = self.index.get(os.path.splitext(os.path.basename(path))[0])
        if idx is None:
            return im_bgr
        s = self.ds.samples[idx]
        has_hand, _, has_roi = (bool(f) for f in s['flags'])
        hb, rb = clip(s['hand_bbox']), clip(s['roi_bbox'] if has_roi else None)
        rgb = np.ascontiguousarray(im_bgr[..., ::-1])
        out, _ = self.ds._maybe_copy_paste(rgb, idx, s, has_hand, hb, rb, has_roi)
        return np.ascontiguousarray(out[..., ::-1])


class PasteDataset(YOLODataset):
    """Pastes at the stored resolution before resizing, as SpatialNet does, so the pasted hands keep
    their relative size. Mosaic, mixup and copy-paste are off, so the image buffer is not needed."""
    paster = None

    def load_image(self, i, *args, **kwargs):
        if self.paster is None:
            return super().load_image(i, *args, **kwargs)
        f = self.im_files[i]
        im = self.paster(f, cv2.imread(f))
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            w, h = min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]


class PasteTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        if mode == 'train':
            ds.__class__ = PasteDataset
            ds.paster = Paster()
            # evidence that training images are pasted: save a few as the training loader sees them
            ex = os.path.join(OUT, 'paste_examples')
            os.makedirs(ex, exist_ok=True)
            for i in range(min(8, len(ds.im_files))):
                cv2.imwrite(os.path.join(ex, os.path.basename(ds.im_files[i])), ds.load_image(i)[0])
            info['paste_trainer'] = f'{type(ds).__name__}, {len(ds.im_files)} images, examples in paste_examples/'
            print('pasting on:', info['paste_trainer'], flush=True)
        return ds


# 3. Initialisation and training
if MODE == 'coco':
    model = YOLO('yolo26n.pt')
    info['init'] = 'yolo26n.pt (COCO)'
else:
    model = YOLO('yolo26n.yaml')
    cls_sd = YOLO('yolo26n-cls.pt').model.state_dict()
    det_sd = model.model.state_dict()
    shared = [k for k, v in cls_sd.items() if k in det_sd and det_sd[k].shape == v.shape]
    model.load('yolo26n-cls.pt')   # transfers every tensor whose name and shape match
    info['init'] = (f'yolo26n-cls.pt (ImageNet): {len(shared)}/{len(det_sd)} tensors transferred, '
                    f'layers {sorted({int(k.split(".")[1]) for k in shared})}')
print('init:', info['init'], flush=True)

t0 = time.time()
model.train(data=data_yaml, imgsz=128, epochs=EPOCHS, patience=50, batch=64, seed=SEED, deterministic=True,
            workers=4, mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0, project=OUT, name='train',
            exist_ok=True, plots=False, device=DEVICE,
            trainer=PasteTrainer if MODE == 'imagenet_paste' else None)
info['train_seconds'] = round(time.time() - t0)
best = os.path.join(OUT, 'train', 'weights', 'best.pt')
res_csv = os.path.join(OUT, 'train', 'results.csv')
if os.path.exists(res_csv):
    with open(res_csv) as f:
        rows = list(csv.DictReader(f))
    info['epochs_run'] = len(rows)
    fit = [0.1 * float(r.get('metrics/mAP50(B)', 0) or 0) + 0.9 * float(r.get('metrics/mAP50-95(B)', 0) or 0)
           for r in rows]
    info['best_epoch'] = int(np.argmax(fit)) + 1 if fit else None


# 4. Pasted test set: identical to SpatialNet's multi-hand evaluation (seed 42)
mh = KeypointPointingDataset(DATA_DIR, 'test', img_size=160, augment=False, synthetic_multihand=True, seed=42)
pasted_dir = '/tmp/pasted_test'
shutil.rmtree(pasted_dir, ignore_errors=True)
os.makedirs(pasted_dir)
keep = set(split_names('test'))
dist_rows = []
for idx, s in enumerate(mh.samples):
    if s['name'] not in keep:
        continue
    img = mh._load_rgb(idx)
    has_hand, _, has_roi = (bool(f) for f in s['flags'])
    hb, rb = clip(s['hand_bbox']), clip(s['roi_bbox'] if has_roi else None)
    out, boxes = mh._maybe_copy_paste(img, idx, s, has_hand, hb, rb, has_roi)
    Image.fromarray(out).save(os.path.join(pasted_dir, s['name'] + '.png'))
    dist_rows.append({'name': s['name'], 'n_distractors': len(boxes),
                      'distractors': json.dumps([[round(float(v), 6) for v in b] for b in boxes])})
with open(os.path.join(OUT, 'test_pasted_distractors.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['name', 'n_distractors', 'distractors'])
    w.writeheader()
    w.writerows(dist_rows)


# 5. Per-image predictions: top hand box and top target box with their confidences
def predict(paths, tag):
    m = YOLO(best)
    rows = []
    for i in range(0, len(paths), 256):
        chunk = paths[i:i + 256]
        for p, r in zip(chunk, m.predict(chunk, imgsz=128, conf=0.001, max_det=50, batch=64,
                                         device=DEVICE, verbose=False)):
            xywhn = r.boxes.xywhn.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            row = {'name': os.path.splitext(os.path.basename(p))[0]}
            for c, key in ((0, 'hand'), (1, 'target')):
                sel = np.where(cls == c)[0]
                if len(sel):
                    j = sel[conf[sel].argmax()]
                    row[f'{key}_conf'] = float(conf[j])
                    row.update({f'{key}_{k}': float(v) for k, v in zip(('cx', 'cy', 'w', 'h'), xywhn[j])})
                else:
                    row[f'{key}_conf'] = 0.0
                    row.update({f'{key}_{k}': '' for k in ('cx', 'cy', 'w', 'h')})
            rows.append(row)
    keys = ['name'] + [f'{k}_{f}' for k in ('hand', 'target') for f in ('conf', 'cx', 'cy', 'w', 'h')]
    with open(os.path.join(OUT, f'predictions_{tag}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows


for split in ('val', 'test'):
    predict([image_path(split, n) for n in split_names(split)], split)
predict([os.path.join(pasted_dir, n + '.png') for n in split_names('test')], 'test_pasted')


# 6. Quick look: target P@0.5 on clean and pasted test (images with a target)
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-8)


gt = {n: parse_annotation(os.path.join(DATA_DIR, 'test', 'labels', n + '.txt')) for n in split_names('test')}
for tag in ('test', 'test_pasted'):
    with open(os.path.join(OUT, f'predictions_{tag}.csv')) as f:
        rows = list(csv.DictReader(f))
    ious = [iou([float(r[f'target_{k}']) for k in ('cx', 'cy', 'w', 'h')], clip(gt[r['name']]['roi']))
            if r['target_cx'] != '' else 0.0 for r in rows if gt[r['name']]['roi'] is not None]
    info[f'{tag}_target_p50'] = round(100 * float(np.mean(np.array(ious) >= 0.5)), 2) if ious else None
for p in ('args.yaml', 'results.csv'):
    if os.path.exists(os.path.join(OUT, 'train', p)):
        shutil.copy(os.path.join(OUT, 'train', p), os.path.join(OUT, p))
with open(os.path.join(OUT, 'info.json'), 'w') as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2), flush=True)
