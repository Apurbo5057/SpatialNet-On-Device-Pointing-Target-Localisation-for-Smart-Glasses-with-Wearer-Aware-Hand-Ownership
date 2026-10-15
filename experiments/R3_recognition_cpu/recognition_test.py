"""Downstream recognition test. Does a recogniser name the labelled object from each model's target crop?

For every test image with a target, crop the box predicted by each model (and the labelled box and the
whole frame as references) from the 320 px copy of the image, restore the original aspect ratio, pad the
crop to a square, and classify it zero-shot over the 348 EgoPoint category names with two off-the-shelf
recognisers (CLIP ViT-L/14 and SigLIP so400m on a GPU). No model is trained. Two crop protocols: the box
itself (tight) and the box enlarged 1.5 times about its centre (context). Without a GPU, the script falls
back to base-size recognisers (CLIP ViT-B/16, SigLIP base) and tight crops only.

Inputs: mosharafapurbo/spatialnet-recognition-inputs (boxes.csv, categories.txt; built by make_inputs.py)
        mosharafapurbo/egopoint-spatialnet-320 (test images)
Output: /kaggle/working/recognition/per_crop_<recogniser>.csv  one row per (protocol, source, image):
        top-1 name, top-5 names, rank of the labelled name (0 = no box)
        /kaggle/working/recognition/info.json
Everything runs in one process (no DataLoader workers): forking workers after the parent has used the
image processor hung an earlier version of this script.
"""
import csv
import json
import math
import os
import time
import traceback

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, __version__ as TRANSFORMERS_VERSION

QUICK = False
OUT = '/kaggle/working/recognition'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':            # primary run
    RECOGNISERS = [('clip_vit_l14', 'openai/clip-vit-large-patch14'),
                   ('siglip_so400m', 'google/siglip-so400m-patch14-384')]
    PROTOCOLS = {'tight': 1.0, 'context': 1.5}
    BATCH = 128
else:                           # CPU fallback: base-size recognisers, tight crops only
    RECOGNISERS = [('clip_vit_b16', 'openai/clip-vit-base-patch16'),
                   ('siglip_b16', 'google/siglip-base-patch16-224')]
    PROTOCOLS = {'tight': 1.0}
    BATCH = 32
    torch.set_num_threads(os.cpu_count() or 4)
TEMPLATES = ['a photo of a {}.', 'a photo of the {}.', 'a cropped photo of a {}.', 'a close-up photo of a {}.']
PAD = (124, 116, 104)          # mean colour used for letterboxing
os.makedirs(OUT, exist_ok=True)
T_START = time.time()


def log(*a):
    print(f'[{time.time() - T_START:7.0f}s]', *a, flush=True)


def find_input(*required):
    for root, dirs, _ in os.walk('/kaggle/input'):
        dirs[:] = [d for d in dirs if d not in ('images', 'labels')]
        if all(os.path.exists(os.path.join(root, r)) for r in required):
            return root
    raise FileNotFoundError(f'No attached input contains {required}')


log('device', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '', 'transformers', TRANSFORMERS_VERSION)
INPUTS = find_input('boxes.csv', 'categories.txt')
IMAGES = os.path.join(find_input('test/images', 'test/labels'), 'test', 'images')
rows = list(csv.DictReader(open(os.path.join(INPUTS, 'boxes.csv'), encoding='utf-8')))
vocab = [c.strip() for c in open(os.path.join(INPUTS, 'categories.txt'), encoding='utf-8') if c.strip()]
SOURCES = [c[:-3] for c in rows[0] if c.endswith('_cx')]          # gt + every model
if QUICK:
    rows = rows[:64]
index = {c: i for i, c in enumerate(vocab)}
assert all(r['category'] in index for r in rows)
true_of = {r['name']: index[r['category']] for r in rows}
log(f'{len(rows)} images, {len(vocab)} categories, sources {SOURCES}, images from {IMAGES}')

# The 320 px copy must carry the same labelled boxes as the files the paper scored.
bad = 0
for r in rows:
    tok = open(os.path.join(IMAGES, '..', 'labels', r['name'] + '.txt')).read().split()
    bad += max(abs(float(a) - float(r[f'gt_{k}'])) for a, k in zip(tok[9:13], ('cx', 'cy', 'w', 'h'))) > 2e-3
log('label mismatches against boxes.csv:', bad)
assert bad == 0
images = {r['name']: Image.open(os.path.join(IMAGES, r['name'] + '.jpg')).convert('RGB') for r in rows}
log('images loaded')


def get_box(r, s):
    v = [r[f'{s}_{k}'] for k in ('cx', 'cy', 'w', 'h')]
    return None if '' in v else [float(x) for x in v]


def crop(img, b, ow, oh, scale):
    """Crop normalised box b (cx, cy, w, h) from the square-squashed image, undo the squash, pad to square."""
    W, H = img.size
    cx, cy, w, h = b[0], b[1], b[2] * scale, b[3] * scale
    x1, x2 = max(0.0, (cx - w / 2) * W), min(float(W), (cx + w / 2) * W)
    y1, y2 = max(0.0, (cy - h / 2) * H), min(float(H), (cy + h / 2) * H)
    if x2 - x1 < 2:                                             # keep at least 2 px
        m = min(max((x1 + x2) / 2, 1.0), W - 1.0)
        x1, x2 = m - 1, m + 1
    if y2 - y1 < 2:
        m = min(max((y1 + y2) / 2, 1.0), H - 1.0)
        y1, y2 = m - 1, m + 1
    c = img.crop((int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))))
    aspect = ((x2 - x1) / W * ow) / ((y2 - y1) / H * oh)          # width / height in the original photo
    area = c.width * c.height
    nw, nh = max(1, round(math.sqrt(area * aspect))), max(1, round(math.sqrt(area / aspect)))
    c = c.resize((nw, nh), Image.BICUBIC)
    s = max(nw, nh)
    sq = Image.new('RGB', (s, s), PAD)
    sq.paste(c, ((s - nw) // 2, (s - nh) // 2))
    return sq


# Every crop to classify: (protocol, source, name, box, original size, scale); the whole frame is source 'whole'.
jobs = []
for r in rows:
    ow, oh = float(r['orig_w']), float(r['orig_h'])
    jobs.append(('whole', 'whole', r['name'], [0.5, 0.5, 1.0, 1.0], ow, oh, 1.0))
    for p, scale in PROTOCOLS.items():
        for s in SOURCES:
            jobs.append((p, s, r['name'], get_box(r, s), ow, oh, scale))
log(len(jobs), 'crops per recogniser')


class Preprocess:
    """The processor's own resize and normalisation for a square input, applied with PIL and numpy.
    For a square crop, CLIP's shortest-side resize plus centre crop and SigLIP's fixed resize are both a
    plain resize to the model's input size."""

    def __init__(self, processor):
        ip = processor.image_processor
        size = ip.crop_size if getattr(ip, 'do_center_crop', False) and getattr(ip, 'crop_size', None) else ip.size
        self.side = self.first(size, 'height', 'shortest_edge')
        self.mean = np.array(ip.image_mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(ip.image_std, dtype=np.float32).reshape(1, 1, 3)

    @staticmethod
    def first(size, *keys):
        """size may be a dict or a SizeDict, depending on the transformers version."""
        for k in keys:
            try:
                v = size[k]
            except (KeyError, TypeError, AttributeError):
                v = getattr(size, k, None)
            if v:
                return int(v)
        raise ValueError(f'no side length in {size}')

    def __call__(self, sq):
        a = np.asarray(sq.resize((self.side, self.side), Image.BICUBIC), dtype=np.float32) / 255.0
        return torch.from_numpy(((a - self.mean) / self.std).transpose(2, 0, 1).copy())


def tokens(processor, texts, hub):
    kw = {'padding': 'max_length', 'max_length': 64} if 'siglip' in hub else {'padding': True}
    tok = processor(text=texts, return_tensors='pt', truncation=True, **kw)
    return {k: v.to(DEVICE) for k, v in tok.items()}


# The full forward pass returns the projected text_embeds and image_embeds that the model's own logits use,
# in every transformers version (get_*_features changed its return type in recent releases).
def encode_text(model, tok, dummy_px):
    return model(**tok, pixel_values=dummy_px).text_embeds.float()


def encode_images(model, px, dummy_tok):
    return model(**dummy_tok, pixel_values=px).image_embeds.float()


def load(hub):
    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
    try:
        model = AutoModel.from_pretrained(hub, dtype=dtype)
    except TypeError:
        model = AutoModel.from_pretrained(hub, torch_dtype=dtype)
    return model.to(DEVICE).eval(), AutoProcessor.from_pretrained(hub)


info = {'transformers': TRANSFORMERS_VERSION, 'torch': torch.__version__, 'device': DEVICE, 'images': len(rows),
        'categories': len(vocab), 'sources': SOURCES, 'protocols': PROTOCOLS, 'templates': TEMPLATES,
        'recognisers': dict(RECOGNISERS), 'seconds': {}, 'preprocess_check': {}, 'errors': {}}
summary = {}
for key, hub in RECOGNISERS:
    t0 = time.time()
    try:
        model, processor = load(hub)
        prep = Preprocess(processor)
        log(key, 'loaded; input side', prep.side)
        with torch.no_grad():
            # Check the manual preprocessing against the official processor on a few real crops.
            diffs = []
            for p, s, name, b, ow, oh, scale in jobs[:8]:
                sq = crop(images[name], b, ow, oh, scale)
                ref = processor(images=sq, return_tensors='pt')['pixel_values'][0]
                diffs.append(float((ref - prep(sq)).abs().mean()))
            info['preprocess_check'][key] = round(max(diffs), 4)
            log(key, 'mean abs difference from the official preprocessing, worst of 8 crops:', max(diffs))
            dummy_px = prep(Image.new('RGB', (64, 64), PAD))[None].to(DEVICE, model.dtype)
            dummy_tok = tokens(processor, ['a photo.'], hub)
            text = []
            for t in TEMPLATES:
                e = encode_text(model, tokens(processor, [t.format(c) for c in vocab], hub), dummy_px)
                text.append(e / e.norm(dim=-1, keepdim=True))
            text = torch.stack(text).mean(0)
            text = text / text.norm(dim=-1, keepdim=True)
            log(key, 'text embeddings', tuple(text.shape))
            results = []
            for start in range(0, len(jobs), BATCH):
                batch = jobs[start:start + BATCH]
                todo = [(j, prep(crop(images[j[2]], j[3], j[4], j[5], j[6]))) for j in batch if j[3] is not None]
                for j in batch:
                    if j[3] is None:
                        results.append([j[0], j[1], j[2], '', '', 0])
                if todo:
                    px = torch.stack([x for _, x in todo]).to(DEVICE, model.dtype)
                    e = encode_images(model, px, dummy_tok)
                    e = e / e.norm(dim=-1, keepdim=True)
                    order = (e @ text.T).argsort(dim=1, descending=True).cpu().tolist()
                    for (j, _), ranked in zip(todo, order):
                        results.append([j[0], j[1], j[2], vocab[ranked[0]], '|'.join(vocab[k] for k in ranked[:5]),
                                        ranked.index(true_of[j[2]]) + 1])
                if (start // BATCH) % 50 == 0:
                    log(f'{key}: {start + len(batch)} of {len(jobs)} crops')
        with open(os.path.join(OUT, f'per_crop_{key}.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['protocol', 'source', 'name', 'top1', 'top5', 'rank'])
            w.writerows(results)
        for p in ['whole'] + list(PROTOCOLS):
            for s in (['whole'] if p == 'whole' else SOURCES):
                rk = [x[5] for x in results if x[0] == p and x[1] == s]
                summary[f'{key}/{p}/{s}'] = {'top1': round(100 * sum(r == 1 for r in rk) / len(rk), 2),
                                             'top5': round(100 * sum(1 <= r <= 5 for r in rk) / len(rk), 2),
                                             'n': len(rk)}
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    except Exception:
        info['errors'][key] = traceback.format_exc()
        log(key, 'FAILED\n' + info['errors'][key])
    info['seconds'][key] = round(time.time() - t0)
    info['summary'] = summary
    json.dump(info, open(os.path.join(OUT, 'info.json'), 'w'), indent=1)
    log(key, 'done in', info['seconds'][key], 's')

log(json.dumps(summary, indent=1))
