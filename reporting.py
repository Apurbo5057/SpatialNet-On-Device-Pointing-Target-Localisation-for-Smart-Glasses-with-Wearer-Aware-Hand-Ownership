"""
Console tables, paper figures, confidence intervals, dataset statistics and
model profiling shared by train.py, test_evaluate.py and the Kaggle notebook.
"""

import os
import io
import csv
import copy
import json
import platform
import warnings

import numpy as np

SCENES = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
SCENE_NAMES = {(0, 0, 0): 'No hand', (1, 0, 0): 'Non-pointing',
               (1, 1, 0): 'Pointing, no target', (1, 1, 1): 'Pointing at target'}

# validated categorical slots 1-3 (blue, orange, aqua), ink and hairline colours
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SEQ_BLUE = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

WIDTH = 96


# console

def banner(title):
    print('\n' + '=' * WIDTH + f'\n  {title}\n' + '=' * WIDTH, flush=True)


def section(title):
    print(f'\n{title}\n' + '-' * len(title), flush=True)


def kv(pairs, indent=2):
    w = max(len(k) for k, _ in pairs)
    for k, v in pairs:
        print(f'{" " * indent}{k:<{w}}  {v}')


def _cell(v):
    if isinstance(v, float):
        return f'{v:.4f}'
    return '' if v is None else str(v)


def format_table(headers, rows, indent=2):
    rows = [[_cell(c) for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(headers)]
    pad = ' ' * indent

    def line(cells, first_left=True):
        out = []
        for i, c in enumerate(cells):
            out.append(c.ljust(w[i]) if i == 0 and first_left else c.rjust(w[i]))
        return pad + '  '.join(out)

    sep = pad + '  '.join('-' * x for x in w)
    return '\n'.join([line([str(h) for h in headers]), sep] + [line(r) for r in rows])


def print_table(headers, rows, indent=2):
    print(format_table(headers, rows, indent), flush=True)


def markdown_table(headers, rows):
    rows = [[_cell(c) for c in r] for r in rows]
    out = ['| ' + ' | '.join(map(str, headers)) + ' |',
           '|' + '|'.join(['---'] + ['---:'] * (len(headers) - 1)) + '|']
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(out)


def fmt_duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h {m:02d}m' if h else (f'{m}m {s:02d}s' if m else f'{s}s')


def pct(x, digits=1):
    return '-' if x is None else f'{100 * x:.{digits}f}'


# files

def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(to_jsonable(obj), f, indent=2)


def write_csv(path, headers, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows([[('' if c is None else c) for c in r] for r in rows])


# statistics

def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def bootstrap_ci(values, stat=np.mean, n_boot=1000, seed=0):
    """95% percentile bootstrap interval of `stat` over `values`."""
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    s = stat(a[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):              # average ranks over ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, s):
    """Step-wise AP over distinct score thresholds (same definition as scikit-learn)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind='mergesort')
    s, y = s[order], y[order]
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[idx]
    prec = tps / (idx + 1)
    rec = tps / tps[-1]
    return float(np.sum((rec - np.r_[0, rec[:-1]]) * prec))


def roc_pr_curves(y, s, n=501):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    t = np.linspace(0, 1, n)
    pred = s[None, :] >= t[:, None]
    tp = (pred & y).sum(1)
    fp = (pred & ~y).sum(1)
    P, N = max(int(y.sum()), 1), max(int((~y).sum()), 1)
    tpr, fpr = tp / P, fp / N
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    return {'threshold': t, 'fpr': fpr, 'tpr': tpr, 'precision': prec, 'recall': tpr}


# environment, data, model

def env_info(device):
    import torch
    info = {'python': platform.python_version(), 'pytorch': torch.__version__,
            'platform': platform.platform(), 'cpu': platform.processor() or platform.machine(),
            'cpu_threads': os.cpu_count(), 'device': str(device)}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['cuda'] = torch.version.cuda
    return info


def dataset_summary(data_dir):
    """Per split: image/label counts, H P T scene counts and fingertip-target gap bins."""
    from annotation import parse_annotation, fingertip_roi_gap, gap_bin
    out = {}
    for split in ['train', 'val', 'test']:
        ldir = os.path.join(data_dir, split, 'labels')
        idir = os.path.join(data_dir, split, 'images')
        if not os.path.isdir(ldir):
            continue
        images = {os.path.splitext(f)[0] for f in os.listdir(idir)} if os.path.isdir(idir) else set()
        scenes = {f: 0 for f in SCENES}
        gaps = {'on-object': 0, 'near': 0, 'far': 0}
        n, missing = 0, 0
        for lf in sorted(f for f in os.listdir(ldir) if f.endswith('.txt')):
            ann = parse_annotation(os.path.join(ldir, lf))
            n += 1
            missing += lf[:-4] not in images
            scenes[ann['flags']] += 1
            if ann['roi'] is not None:
                gaps[gap_bin(fingertip_roi_gap(ann['fingertip'], ann['roi']))] += 1
        out[split] = {'labels': n, 'images_without_label': len(images) - (n - missing),
                      'labels_without_image': missing,
                      'scenes': {SCENE_NAMES[f]: c for f, c in scenes.items()},
                      'target_gap': gaps}
    return out


def print_dataset_summary(summary):
    names = [SCENE_NAMES[f] for f in SCENES]
    rows = []
    for split, s in summary.items():
        rows.append([split, s['labels']] + [s['scenes'][k] for k in names]
                    + [s['target_gap'][b] for b in ['on-object', 'near', 'far']])
    if len(rows) > 1:
        rows.append(['total'] + [sum(r[i] for r in rows) for i in range(1, len(rows[0]))])
    print_table(['Split', 'Images'] + names + ['Tgt on-object', 'Tgt near', 'Tgt far'], rows)
    bad = {k: (v['labels_without_image'], v['images_without_label']) for k, v in summary.items()
           if v['labels_without_image'] or v['images_without_label']}
    for split, (a, b) in bad.items():
        print(f'  [WARN] {split}: {a} labels without an image, {b} images without a label')


def model_profile(model, img_size):
    """Parameter counts, FP32 size and forward FLOPs at batch size 1 (on a CPU copy)."""
    import torch
    m = copy.deepcopy(model).cpu().eval()
    total = sum(p.numel() for p in m.parameters())
    backbone = sum(p.numel() for p in m.get_backbone_params())
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(torch.zeros(1, 3, img_size, img_size))
        flops = int(fc.get_total_flops())
    except Exception:
        pass
    return {'params_total': total, 'params_backbone': backbone, 'params_heads': total - backbone,
            'size_fp32_mb': total * 4 / 1024 ** 2, 'gflops': None if flops is None else flops / 1e9,
            'input': f'1x3x{img_size}x{img_size}'}


def serialized_sizes_kb(model):
    """On-disk size of the FP32, FP16 and dynamic-INT8 state dicts."""
    import torch
    import torch.nn as nn
    m = copy.deepcopy(model).cpu().eval()

    def size(sd):
        buf = io.BytesIO()
        torch.save(sd, buf)
        return buf.getbuffer().nbytes / 1024

    out = {'fp32': size(m.state_dict()),
           'fp16': size({k: (v.half() if v.is_floating_point() else v) for k, v in m.state_dict().items()})}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out['int8'] = size(torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8).state_dict())
    return out


def measure_latency(model, img_size, device='cpu', threads=None, warmup=10, runs=50):
    """Median and p90 forward latency (ms) at batch size 1."""
    import time
    import torch
    m = copy.deepcopy(model).to(device).eval()
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    prev = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        times = []
        with torch.no_grad():
            for i in range(warmup + runs):
                if device != 'cpu':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                m(x)
                if device != 'cpu':
                    torch.cuda.synchronize()
                if i >= warmup:
                    times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(prev)
    return {'median_ms': float(np.median(times)), 'p90_ms': float(np.percentile(times, 90)),
            'fps': float(1000 / np.median(times))}


# figures

def paper_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.6, 'axes.labelcolor': INK2,
        'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'lines.linewidth': 1.4, 'lines.markersize': 3.5, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    return plt


def save_figure(fig, path_stem):
    """Save as PDF (vector, for the paper) and PNG (preview)."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path_stem) or '.', exist_ok=True)
    fig.savefig(path_stem + '.pdf', bbox_inches='tight')
    fig.savefig(path_stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
