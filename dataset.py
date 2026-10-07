"""
Dataset and dataloaders for SpatialNet: H P T flags, hand box, fingertip, ROI box
(label format: docs/annotation_format.md), with distractor-hand copy-paste
augmentation and a deterministic synthetic multi-hand eval split.
Imported by train.py and test_evaluate.py via get_dataloaders / get_multihand_loader.
"""

import os
import math
import random
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import config as cfg
from annotation import parse_annotation, fingertip_roi_gap, gap_bin  # noqa: F401 (re-exported)

# no online version check, and no deprecation notice for ShiftScaleRotate (kept for identical augmentation)
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
warnings.filterwarnings('ignore', message='.*ShiftScaleRotate is a special case of Affine.*')
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    ALBUMENTATIONS_AVAILABLE = False

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HAND_LABEL, ROI_LABEL, DISTRACTOR_LABEL = 0, 2, 9


def _box_to_xyxy(b):
    return b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2


def _inter_area(a, b):
    ax1, ay1, ax2, ay2 = _box_to_xyxy(a)
    bx1, by1, bx2, by2 = _box_to_xyxy(b)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def _feathered_alpha(h, w, feather_frac):
    """Alpha mask in [0,1] with a cosine ramp on all four borders."""
    f = max(1, int(round(feather_frac * min(h, w))))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 2 * f + 1)[1:f + 1])   # 0..1
    ax = np.ones(w, dtype=np.float32)
    ay = np.ones(h, dtype=np.float32)
    ax[:f] = ramp[:min(f, w)]
    ax[w - f:] = ramp[:min(f, w)][::-1]
    ay[:f] = ramp[:min(f, h)]
    ay[h - f:] = ramp[:min(f, h)][::-1]
    return (ay[:, None] * ax[None, :])[..., None]


def _crop_box(img, box, margin=0.08):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = _box_to_xyxy(box)
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, (x1 - mw) * W)), int(max(0, (y1 - mh) * H))
    x2, y2 = int(min(W, (x2 + mw) * W)), int(min(H, (y2 + mh) * H))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]


def paste_patch(img, patch, cx, cy, feather_frac, rng, jitter=True):
    """Alpha-paste `patch` centred at normalised (cx,cy); returns its cx,cy,w,h box or None."""
    H, W = img.shape[:2]
    ph, pw = patch.shape[:2]
    x1 = int(round(cx * W - pw / 2))
    y1 = int(round(cy * H - ph / 2))
    x1 = min(max(0, x1), W - pw)
    y1 = min(max(0, y1), H - ph)
    if x1 < 0 or y1 < 0:
        return None
    region = img[y1:y1 + ph, x1:x1 + pw].astype(np.float32)
    p = patch.astype(np.float32)
    if jitter:
        p = p * rng.uniform(0.85, 1.15) * rng.uniform(0.95, 1.05, size=(1, 1, 3))
        p = np.clip(p, 0, 255)
    a = _feathered_alpha(ph, pw, feather_frac) * rng.uniform(0.9, 1.0)
    img[y1:y1 + ph, x1:x1 + pw] = (a * p + (1 - a) * region).astype(np.uint8)
    return [(x1 + pw / 2) / W, (y1 + ph / 2) / H, pw / W, ph / H]


def _resize_patch(patch, scale, max_frac, img_shape):
    H, W = img_shape[:2]
    ph, pw = patch.shape[:2]
    s = scale
    s = min(s, max_frac * W / pw, max_frac * H / ph)
    nw, nh = max(8, int(pw * s)), max(8, int(ph * s))
    return np.array(Image.fromarray(patch).resize((nw, nh), Image.BILINEAR))


def _reorient(patch, rng):
    """Rotate by 90/180/270 deg and maybe flip so the forearm does not enter from the bottom."""
    k = int(rng.randint(1, 4))
    patch = np.rot90(patch, k)
    if rng.rand() < 0.5:
        patch = patch[:, ::-1]
    return np.ascontiguousarray(patch)


def copy_paste_distractors(img, avoid_boxes, crops, rng, num, decoy_donor=None):
    """Paste up to `num` uint8 hand crops into `img`; returns (img, distractor cx,cy,w,h boxes)."""
    H, W = img.shape[:2]
    boxes = []
    for crop in crops[:num]:
        if crop is None or crop.size == 0:
            continue
        if rng.rand() < cfg.CP_REORIENT_PROB:
            crop = _reorient(crop, rng)
        crop = _resize_patch(crop, rng.uniform(*cfg.CP_SCALE), 0.6, img.shape)
        ph, pw = crop.shape[:2]
        placed = None
        for _ in range(20):
            cx = rng.uniform(pw / W / 2 + 0.02, 1 - pw / W / 2 - 0.02) if pw / W < 0.96 else 0.5
            cy = rng.uniform(ph / H / 2 + 0.02, max(cfg.CP_TOP_FRACTION, ph / H / 2 + 0.03))
            cand = [cx, cy, pw / W, ph / H]
            area = cand[2] * cand[3] + 1e-8
            ok = all(_inter_area(cand, b) / area < cfg.CP_MAX_OVERLAP for b in avoid_boxes)
            ok = ok and all(_inter_area(cand, b) / area < 0.3 for b in boxes)
            if ok:
                placed = cand
                break
        if placed is None:
            continue
        box = paste_patch(img, crop, placed[0], placed[1], cfg.CP_FEATHER, rng)
        if box is not None:
            boxes.append(box)

    if decoy_donor is not None:
        # unlabeled non-hand patch so a pasted rectangle is not itself a hand cue
        # (the caller already drew the CP_DECOY_PROB chance before loading the donor)
        dH, dW = decoy_donor.shape[:2]
        s = rng.uniform(0.15, 0.35)
        pw, ph = int(s * dW), int(s * rng.uniform(0.7, 1.4) * dH)
        pw, ph = max(8, min(pw, dW)), max(8, min(ph, dH))
        x1, y1 = rng.randint(0, dW - pw + 1), rng.randint(0, dH - ph + 1)
        patch = decoy_donor[y1:y1 + ph, x1:x1 + pw]
        cand_c = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        cand = [cand_c[0], cand_c[1], pw / W, ph / H]
        area = cand[2] * cand[3] + 1e-8
        if all(_inter_area(cand, b) / area < cfg.CP_MAX_OVERLAP for b in avoid_boxes):
            paste_patch(img, patch, cand_c[0], cand_c[1], cfg.CP_FEATHER, rng)
    return img, boxes


class KeypointPointingDataset(Dataset):
    """Wearer-hand pointing dataset: H P T flags, hand box, fingertip, ROI box."""

    def __init__(self, data_dir, split='train', img_size=None, augment=True,
                 copy_paste=None, synthetic_multihand=False, seed=42):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size or cfg.IMG_SIZE
        self.augment = augment and split == 'train'
        self.synthetic_multihand = synthetic_multihand
        self.seed = seed
        if copy_paste is None:
            copy_paste = cfg.CP_PROB if split == 'train' else 0.0
        self.cp_prob = float(copy_paste) if self.augment else 0.0

        self.images_dir = os.path.join(data_dir, split, 'images')
        self.labels_dir = os.path.join(data_dir, split, 'labels')

        self.samples = self._load_samples()
        self._crop_cache = {}
        self._crop_cache_limit = max(64, int(getattr(cfg, 'CP_DONOR_POOL', 512)))
        self.donor_idx = [i for i, s in enumerate(self.samples) if s['hand_bbox'] is not None]
        self.pointing_donor_idx = [i for i, s in enumerate(self.samples) if s['flags'][1]]
        self.nohand_idx = [i for i, s in enumerate(self.samples) if not s['flags'][0]]
        self.transform = self._get_transforms()
        self.eval_transform = self._get_eval_transform()
        self._donor_pool = []
        if (self.cp_prob > 0 or self.synthetic_multihand) and self.donor_idx:
            self._prewarm_donor_pool()

    def _load_samples(self):
        samples = []
        label_files = sorted(f for f in os.listdir(self.labels_dir) if f.endswith('.txt'))
        for lf in label_files:
            parsed = parse_annotation(os.path.join(self.labels_dir, lf))
            bn = lf[:-4]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                ip = os.path.join(self.images_dir, bn + ext)
                if os.path.exists(ip):
                    samples.append({
                        'name': bn,
                        'image_path': ip,
                        'hand_bbox': parsed['hand'],
                        'fingertip_keypoint': parsed['fingertip'],
                        'roi_bbox': parsed['roi'],
                        'flags': parsed['flags'],
                    })
                    break
        return samples

    @staticmethod
    def _clip_bbox(b):
        if b is None:
            return [0., 0., 0., 0.]
        return [max(0, min(1, b[0])), max(0, min(1, b[1])),
                max(0.001, min(1, b[2])), max(0.001, min(1, b[3]))]

    @staticmethod
    def _clip_kp(k):
        if k is None:
            return [0., 0.]
        return [max(0, min(1, k[0])), max(0, min(1, k[1]))]

    @staticmethod
    def _compute_dir(hb, kp, roi=None):
        """Unit direction fingertip -> ROI centre, or hand centre -> fingertip without a target."""
        if kp is None or sum(kp) == 0:
            return [0., 0.], 0.
        if roi is not None and sum(roi) > 0:
            dx, dy = roi[0] - kp[0], roi[1] - kp[1]
        elif hb is not None and sum(hb) > 0:
            dx, dy = kp[0] - hb[0], kp[1] - hb[1]
        else:
            return [0., 0.], 0.
        mag = math.sqrt(dx * dx + dy * dy) + 1e-8
        return [dx / mag, dy / mag], math.atan2(dy, dx)

    def _get_transforms(self):
        if not self.augment:
            return None
        if not ALBUMENTATIONS_AVAILABLE:
            raise RuntimeError(
                "albumentations is required to train with augmentation but is not "
                "installed (`pip install albumentations`). Training would silently "
                "fall back to horizontal-flip-only. Pass augment=False if you "
                "genuinely want an unaugmented run.")
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            # no VerticalFlip: a wearer hand entering from the top conflicts with wearer ownership
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                               border_mode=0, p=0.4),
            A.Perspective(scale=(0.02, 0.05), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
            ToTensorV2()
        ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                                     min_visibility=0.3))

    def _get_eval_transform(self):
        if not ALBUMENTATIONS_AVAILABLE:
            return None
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
            ToTensorV2()
        ])

    def _np_to_tensor(self, img):
        img = np.array(Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        img = (img - MEAN) / STD
        return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

    def _load_rgb(self, idx):
        return np.array(Image.open(self.samples[idx]['image_path']).convert('RGB'))

    def _prewarm_donor_pool(self):
        """Decode a bounded set of donor hand crops at startup so epoch 1 avoids per-sample decodes."""
        n = min(int(getattr(cfg, 'CP_DONOR_POOL', 512)), len(self.donor_idx))
        rng = np.random.RandomState(self.seed)
        # keep pointing hands over-represented: they are the hard distractors
        pool = list(self.pointing_donor_idx)[:n // 2]
        rest = [i for i in self.donor_idx if i not in set(pool)]
        if rest:
            take = min(n - len(pool), len(rest))
            pool += [int(rest[j]) for j in rng.choice(len(rest), size=take, replace=False)]
        for j in pool:
            crop = _crop_box(self._load_rgb(j), self.samples[j]['hand_bbox'])
            if crop is not None:
                self._crop_cache[j] = crop
                self._donor_pool.append(j)

    def _donor_crop(self, idx):
        """Hand crop of donor `idx`, memoised up to the cache limit."""
        hit = self._crop_cache.get(idx)
        if hit is not None:
            return hit
        crop = _crop_box(self._load_rgb(idx), self.samples[idx]['hand_bbox'])
        if crop is not None and len(self._crop_cache) < self._crop_cache_limit:
            self._crop_cache[idx] = crop
        return crop

    def _maybe_copy_paste(self, img, idx, s, has_hand, hb, rb, has_roi):
        """Returns (img, distractor boxes)."""
        if self.synthetic_multihand:
            rng = np.random.RandomState(self.seed * 100003 + idx)
            do = True
            num = int(rng.randint(1, cfg.CP_MAX_DISTRACTORS + 1))
        else:
            if self.cp_prob <= 0:
                return img, []
            rng = np.random.RandomState(random.randint(0, 2 ** 31 - 1))
            p = self.cp_prob if has_hand else self.cp_prob * (cfg.CP_ON_NOHAND_PROB / max(cfg.CP_PROB, 1e-8))
            do = rng.rand() < p
            num = int(rng.randint(1, cfg.CP_MAX_DISTRACTORS + 1))
        if not do or not self.donor_idx:
            return img, []

        donors = []
        for _ in range(num):
            # self-paste: identical appearance, so only position/orientation/scale mark the wearer
            if has_hand and rng.rand() < cfg.CP_SELF_PASTE_PROB:
                donors.append(_crop_box(img, hb))
                continue
            pool = self._donor_pool or self.donor_idx
            j = int(pool[rng.randint(len(pool))])
            if j == idx:
                continue
            donors.append(self._donor_crop(j))
        donors = [c for c in donors if c is not None]
        avoid = []
        if has_hand:
            avoid.append(hb)
        if has_roi:
            avoid.append(rb)
        decoy = None
        if (self.nohand_idx and not self.synthetic_multihand
                and rng.rand() < cfg.CP_DECOY_PROB):
            decoy = self._load_rgb(int(self.nohand_idx[rng.randint(len(self.nohand_idx))]))
        img = np.ascontiguousarray(img)
        return copy_paste_distractors(img, avoid, donors, rng, num, decoy)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self._load_rgb(idx)
        h, w = img.shape[:2]

        has_hand, has_kp, has_roi = (bool(f) for f in s['flags'])   # H, P, T

        hb = self._clip_bbox(s['hand_bbox'])
        kp = self._clip_kp(s['fingertip_keypoint'])
        rb = self._clip_bbox(s['roi_bbox'] if has_roi else None)
        gap = fingertip_roi_gap(kp, rb) if has_roi else -1.0

        img, distractors = self._maybe_copy_paste(img, idx, s, has_hand, hb, rb, has_roi)

        if self.augment and self.transform is not None:
            try:
                bboxes, labels = [], []
                if has_hand:
                    bboxes.append(hb); labels.append(HAND_LABEL)
                if has_roi:
                    bboxes.append(rb); labels.append(ROI_LABEL)
                for d in distractors:
                    bboxes.append(self._clip_bbox(d)); labels.append(DISTRACTOR_LABEL)
                kp_px = [(kp[0] * w, kp[1] * h)] if has_kp else []
                t = self.transform(image=img, keypoints=kp_px, bboxes=bboxes, class_labels=labels)
                out_boxes = {}
                new_distractors = []
                for b, l in zip(t['bboxes'], t['class_labels']):
                    if l == DISTRACTOR_LABEL:
                        new_distractors.append(list(b))
                    else:
                        out_boxes[l] = list(b)
                if (has_hand and HAND_LABEL not in out_boxes) or (has_roi and ROI_LABEL not in out_boxes):
                    raise RuntimeError('box dropped by augmentation')
                img_t = t['image']
                if has_hand:
                    hb = out_boxes[HAND_LABEL]
                if has_roi:
                    rb = out_boxes[ROI_LABEL]
                if has_kp and len(t['keypoints']) > 0:
                    kp = [t['keypoints'][0][0] / self.img_size, t['keypoints'][0][1] / self.img_size]
                distractors = new_distractors
            except Exception:
                img_t = self.eval_transform(image=img)['image']
        elif self.augment and self.transform is None:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                if has_hand:
                    hb[0] = 1 - hb[0]
                if has_roi:
                    rb[0] = 1 - rb[0]
                if has_kp:
                    kp[0] = 1 - kp[0]
                distractors = [[1 - d[0], d[1], d[2], d[3]] for d in distractors]
            img_t = self._np_to_tensor(img)
        else:
            img_t = self.eval_transform(image=img)['image'] if self.eval_transform is not None \
                else self._np_to_tensor(img)

        roi_center = [rb[0], rb[1]] if has_roi else None
        dv, ang = self._compute_dir(hb if has_hand else None, kp if has_kp else None, roi_center)

        K = cfg.CP_MAX_DISTRACTORS
        dboxes = torch.zeros(K, 4, dtype=torch.float32)
        for i, d in enumerate(distractors[:K]):
            dboxes[i] = torch.tensor(self._clip_bbox(d), dtype=torch.float32)

        return {
            'image': img_t,
            'hand_bbox': torch.tensor(hb, dtype=torch.float32),
            'fingertip_kp': torch.tensor(kp, dtype=torch.float32),
            'pointing_dir': torch.tensor(dv, dtype=torch.float32),
            'pointing_angle': torch.tensor([ang], dtype=torch.float32),
            'sin_cos_angle': torch.tensor([math.sin(ang), math.cos(ang)], dtype=torch.float32),
            'roi_bbox': torch.tensor(rb, dtype=torch.float32),
            'valid_mask': torch.tensor([float(has_hand), float(has_kp), float(has_roi)]),
            'hand_conf': torch.tensor([float(has_hand)]),    # H
            'point_conf': torch.tensor([float(has_kp)]),     # P
            'roi_conf': torch.tensor([float(has_roi)]),      # T
            'distractor_boxes': dboxes,
            'num_distractors': torch.tensor([float(min(len(distractors), K))]),
            'ft_roi_gap': torch.tensor([float(gap)]),
            'index': torch.tensor([float(idx)]),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def get_dataloaders(data_dir, img_size=None, batch_size=32, num_workers=2, copy_paste=None):
    """Train (augmented, shuffled), val and test dataloaders."""
    train_ds = KeypointPointingDataset(data_dir, 'train', img_size, True, copy_paste=copy_paste)
    val_ds = KeypointPointingDataset(data_dir, 'val', img_size, False)
    test_ds = KeypointPointingDataset(data_dir, 'test', img_size, False)

    args = {'batch_size': batch_size, 'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(), 'collate_fn': collate_fn}
    train_dl = DataLoader(train_ds, shuffle=True, **args)

    return (train_dl,
            DataLoader(val_ds, shuffle=False, **args),
            DataLoader(test_ds, shuffle=False, **args))


def get_eval_loader(data_dir, split, img_size=None, batch_size=32, num_workers=0):
    """Unaugmented loader over one split (no training set is built)."""
    ds = KeypointPointingDataset(data_dir, split, img_size, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)


def get_multihand_loader(data_dir, split='test', img_size=None, batch_size=32,
                         num_workers=0, seed=42):
    """Loader over a split with deterministic synthetic distractor hands (ownership eval)."""
    ds = KeypointPointingDataset(data_dir, split, img_size, augment=False,
                                 synthetic_multihand=True, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)


def mixup_data(images, batch, alpha=0.2):
    """MixUp within same-H P T groups (images only; labels unmixed)."""
    if alpha <= 0:
        return images, batch, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    bs = images.size(0)
    group = (4 * batch['hand_conf'] + 2 * batch['point_conf'] + batch['roi_conf'])
    group = group.view(-1).detach().cpu().numpy()
    index = np.arange(bs)
    for g in np.unique(group):
        idxs = np.where(group == g)[0]
        index[idxs] = np.random.permutation(idxs)
    index = torch.from_numpy(index).long().to(images.device)
    return lam * images + (1 - lam) * images[index], batch, lam


if __name__ == "__main__":
    print("Dataset module loaded successfully!")
    print(f"Albumentations available: {ALBUMENTATIONS_AVAILABLE}")
