"""
Train SpatialNet and save checkpoints plus the training statistics and figures used in the paper.
Usage: python train.py --data_dir <dataset> --output_dir <out> [--quantize] [--resume <out>/last_checkpoint.pth]
Writes checkpoints to <out>/ and history, curves and run statistics to <out>/training/.
"""

import os
import sys
import time
import json
import argparse
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import defaultdict

from model import SpatialNet
from dataset import get_dataloaders, mixup_data
from loss import MultiTaskLoss
import config as cfg
import reporting as R
from utils import save_quantized_models

# AMP may skip the first optimizer steps while the grad scale calibrates; the scheduler warning is expected
warnings.filterwarnings('ignore', message=r'Detected call of `lr_scheduler\.step\(\)` before')

# stdout is a pipe under subprocess (e.g. Kaggle); line-buffer so progress shows live
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# per-epoch quantities kept in history and plotted (key in loss metrics -> label)
TRACKED = {'hand_iou': 'Hand IoU', 'roi_iou': 'Target IoU', 'ft_px_err': 'Fingertip error (px)',
           'angle_err_deg': 'Ray angle error (deg)', 'hand_conf_acc': 'Hand gate acc',
           'point_conf_acc': 'Pointing gate acc', 'roi_conf_acc': 'Target gate acc',
           'hand_fn_rate': 'Missed hand rate', 'point_fn_rate': 'Missed pointing rate',
           'roi_fn_rate': 'Missed target rate', 'ownership_acc': 'Ownership acc'}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_param_groups(model, backbone_lr_mult=0.1, base_lr=3e-3):
    return [
        {'params': list(model.get_backbone_params()), 'lr': base_lr * backbone_lr_mult},
        {'params': list(model.get_head_params()), 'lr': base_lr},
    ]


def build_model(args):
    return SpatialNet(
        input_size=args.img_size, load_pretrained=args.load_pretrained,
        use_bifpn=not args.no_bifpn, use_coordconv=not args.no_coordconv,
        mixer_depth=args.mixer_depth, anchor_chain=not args.no_anchor_chain,
        local_softargmax=not args.global_softargmax,
        roi_geometry=args.roi_geometry, ft_box_prior=not args.no_ft_box_prior)


def make_anchors(batch, p):
    """Scheduled anchor forcing: per sample and anchor, feed the GT location
    (with probability p, only where GT is valid) instead of the prediction."""
    if p <= 0:
        return None
    valid = batch['valid_mask']
    hand_v, ft_v, roi_v = valid[:, 0:1], valid[:, 1:2], valid[:, 2:3]

    def mix(v):
        return (torch.rand_like(v) < p).to(v.dtype) * v

    return {'hand': (batch['hand_bbox'][:, :2], mix(hand_v)),
            'fingertip': (batch['fingertip_kp'], mix(ft_v)),
            'roi': (batch['roi_bbox'][:, :2], mix(roi_v))}


def forcing_prob(epoch, p0, warm_epochs):
    """Forcing probability decaying linearly from p0 to zero at `warm_epochs`."""
    if p0 <= 0 or warm_epochs <= 0:
        return 0.0
    return float(max(0.0, p0 * (1.0 - epoch / warm_epochs)))


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, mixup_alpha=0.2, grad_clip=1.0, use_amp=True,
                    force_p=0.0, epoch=0, log_every=0):
    model.train()
    epoch_loss, num_batches = 0.0, 0
    epoch_metrics = defaultdict(float)
    n_batches = len(loader)
    t_start = time.time()
    for batch in loader:
        batch_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        images = batch_device['image']
        if mixup_alpha > 0:
            images, batch_device, _ = mixup_data(images, batch_device, mixup_alpha)
        anchors = make_anchors(batch_device, force_p)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, anchors=anchors)
            loss, metrics = criterion(preds, batch_device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        epoch_loss += loss.item()
        for k, v in metrics.items():
            epoch_metrics[k] += v
        num_batches += 1
        if log_every and num_batches % log_every == 0:
            rate = num_batches / max(time.time() - t_start, 1e-6)
            print(f'    epoch {epoch + 1} | batch {num_batches}/{n_batches} | '
                  f'loss {epoch_loss / num_batches:.4f} | eta {(n_batches - num_batches) / rate / 60:.1f} min',
                  flush=True)
    n = max(num_batches, 1)
    return epoch_loss / n, {k: v / n for k, v in epoch_metrics.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=True):
    model.eval()
    val_loss, num_batches = 0.0, 0
    val_metrics = defaultdict(float)
    for batch in loader:
        batch_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(batch_device['image'])
            loss, metrics = criterion(preds, batch_device)
        val_loss += loss.item()
        for k, v in metrics.items():
            val_metrics[k] += v
        num_batches += 1
    n = max(num_batches, 1)
    return val_loss / n, {k: v / n for k, v in val_metrics.items()}


# console layout of the per-epoch table
EPOCH_HEADERS = ['Epoch', 'Time', 'LR', 'Train loss', 'Val loss', 'Hand IoU', 'Target IoU',
                 'Tip err px', 'Ray err deg', 'Gate acc % H/P/T', 'Missed % H/P/T', 'Saved']
EPOCH_WIDTHS = [7, 6, 8, 10, 9, 8, 10, 10, 11, 17, 15, 12]


def _row(cells):
    return '  ' + '  '.join(str(c).rjust(w) if i else str(c).ljust(w)
                            for i, (c, w) in enumerate(zip(cells, EPOCH_WIDTHS)))


def print_epoch_header():
    print(_row(EPOCH_HEADERS))
    print('  ' + '  '.join('-' * w for w in EPOCH_WIDTHS), flush=True)


def print_epoch_row(epoch, epochs, dt, lr, tr_loss, val_loss, vm, saved):
    tri = lambda a, b, c: f'{100 * a:5.1f}/{100 * b:5.1f}/{100 * c:5.1f}'
    print(_row([f'{epoch}/{epochs}', f'{dt:.0f}s', f'{lr:.2e}', f'{tr_loss:.4f}', f'{val_loss:.4f}',
                f'{vm["hand_iou"]:.3f}', f'{vm["roi_iou"]:.3f}', f'{vm["ft_px_err"]:.2f}',
                f'{vm["angle_err_deg"]:.1f}',
                tri(vm['hand_conf_acc'], vm['point_conf_acc'], vm['roi_conf_acc']),
                tri(vm['hand_fn_rate'], vm['point_fn_rate'], vm['roi_fn_rate']), saved]), flush=True)


def plot_training_curves(history, best_epoch, tdir):
    """One PDF/PNG per panel (for the paper) plus a combined overview."""
    plt = R.paper_style()
    from matplotlib.ticker import MaxNLocator
    ep = np.array(history['epoch'])
    panels = [
        ('loss', 'Loss', [('train_loss', 'Train'), ('val_loss', 'Validation')], None),
        ('iou', 'IoU (validation)', [('val_hand_iou', 'Hand'), ('val_roi_iou', 'Target')], (0, 1)),
        ('fingertip_error', 'Fingertip error (px)', [('train_ft_px_err', 'Train'), ('val_ft_px_err', 'Validation')], None),
        ('ray_angle_error', 'Ray angle error (deg)', [('train_angle_err_deg', 'Train'),
                                                      ('val_angle_err_deg', 'Validation')], None),
        ('gate_accuracy', 'Gate accuracy (validation)', [('val_hand_conf_acc', 'Hand'), ('val_point_conf_acc', 'Pointing'),
                                                          ('val_roi_conf_acc', 'Target')], None),
        ('learning_rate', 'Learning rate (heads)', [('lr', None)], None),
    ]
    colors, markers = [R.C1, R.C2, R.C3], ['o', 's', '^']

    def draw(ax, title, series, ylim):
        for i, (key, label) in enumerate(series):
            if key not in history:
                continue
            ax.plot(ep, history[key], color=colors[i], label=label, marker=markers[i],
                    markevery=max(1, len(ep) // 12))
        if best_epoch:
            ax.axvline(best_epoch, color=R.MUTED, lw=0.8)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ylim:
            ax.set_ylim(*ylim)
        if len(series) > 1:
            ax.legend()

    for name, title, series, ylim in panels:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        draw(ax, title, series, ylim)
        R.save_figure(fig, os.path.join(tdir, 'figures', f'curve_{name}'))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, (_, title, series, ylim) in zip(axes.flat, panels):
        draw(ax, title, series, ylim)
    fig.suptitle(f'SpatialNet training (vertical line: best epoch {best_epoch})', x=0.01, ha='left',
                 fontsize=9, color=R.INK)
    fig.tight_layout()
    R.save_figure(fig, os.path.join(tdir, 'figures', 'training_curves'))


def save_history(history, tdir):
    keys = list(history.keys())
    R.write_csv(os.path.join(tdir, 'history.csv'), keys, list(zip(*[history[k] for k in keys])))
    R.write_json(os.path.join(tdir, 'history.json'), history)


def write_run_summary(info, tdir):
    """run_info.json (everything) and summary.md (paper-ready tables)."""
    R.write_json(os.path.join(tdir, 'run_info.json'), info)
    md = ['# SpatialNet training run', '']
    md += ['## Setup', '', R.markdown_table(['Setting', 'Value'], info['setup_table']), '']
    md += ['## Dataset', '', R.markdown_table(info['dataset_table']['headers'], info['dataset_table']['rows']), '']
    md += ['## Model', '', R.markdown_table(['Property', 'Value'], info['model_table']), '']
    if 'result_table' in info:
        md += ['## Result', '', R.markdown_table(['', 'Value'], info['result_table']), '']
        md += ['## Validation metrics at the best epoch', '',
               R.markdown_table(['Metric', 'Value'], info['best_epoch_table']), '']
    if 'quantized_table' in info:
        md += ['## Exported model sizes', '', R.markdown_table(['Format', 'Size (KB)', 'Budget used'],
                                                               info['quantized_table']), '']
    with open(os.path.join(tdir, 'summary.md'), 'w') as f:
        f.write('\n'.join(md))


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and not args.no_amp
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tdir = os.path.join(args.output_dir, 'training')
    os.makedirs(tdir, exist_ok=True)
    run_start = time.time()

    R.banner('SpatialNet training')
    env = R.env_info(device)
    R.section('Environment')
    R.kv([('Device', env.get('gpu', env['cpu']) + f'  ({device}, mixed precision {"on" if use_amp else "off"})'),
          ('PyTorch / Python', f'{env["pytorch"]} / {env["python"]}'), ('Seed', args.seed),
          ('Data', args.data_dir), ('Output', args.output_dir)])

    R.section('Dataset')
    data_summary = R.dataset_summary(args.data_dir)
    R.print_dataset_summary(data_summary)
    train_dl, val_dl, _ = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers,
        copy_paste=args.copy_paste)

    model = build_model(args)
    profile = R.model_profile(model, args.img_size)
    pretrained = getattr(model.backbone, 'pretrained_path', None)
    model = model.to(device)
    R.section('Model')
    model_table = [
        ('Backbone', 'ShuffleNetV2, ' + ('ImageNet-pretrained' if pretrained else 'random init (no pretrained weights found)')),
        ('Parameters', f'{profile["params_total"]:,} (backbone {profile["params_backbone"]:,}, '
                       f'heads {profile["params_heads"]:,})'),
        ('Size (FP32)', f'{profile["size_fp32_mb"]:.2f} MB'),
        ('Compute', f'{profile["gflops"]:.3f} GFLOPs per {args.img_size}x{args.img_size} image'
         if profile['gflops'] is not None else 'n/a'),
    ]
    R.kv(model_table)

    optimizer = optim.AdamW(get_param_groups(model, args.backbone_lr_mult, args.lr), weight_decay=1e-4)
    total_steps = max(1, args.epochs * len(train_dl))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr * args.backbone_lr_mult, args.lr],
        total_steps=total_steps, pct_start=0.1, anneal_strategy='cos',
        div_factor=25.0, final_div_factor=1000.0)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    weights = {k: getattr(args, f'w_{k}') for k in cfg.LOSS_WEIGHTS}
    criterion = MultiTaskLoss(weights=weights, focal_gamma=args.focal_gamma,
                              fn_penalty_mult=args.fn_penalty,
                              hand_fn_penalty_mult=args.hand_fn_penalty,
                              ownership_neg_weight=args.ownership_weight)

    setup_table = [
        ('Input size', f'{args.img_size} x {args.img_size}'),
        ('Epochs (max) / early-stop patience', f'{args.epochs} / {args.patience} epochs on val loss'),
        ('Batch size / steps per epoch', f'{args.batch_size} / {len(train_dl)}'),
        ('Optimizer', 'AdamW, weight decay 1e-4'),
        ('Learning rate', f'one-cycle, peak {args.lr:g} (backbone x{args.backbone_lr_mult:g}), '
                          f'10% warm-up, cosine decay'),
        ('Gradient clipping', f'{args.grad_clip:g}'),
        ('MixUp', f'alpha {args.mixup_alpha:g}, within same H P T group' if args.mixup_alpha > 0 else 'off'),
        ('Distractor-hand copy-paste', f'p {args.copy_paste:g} (no-hand p {cfg.CP_ON_NOHAND_PROB:g}), '
                                       f'up to {cfg.CP_MAX_DISTRACTORS} hands' if args.copy_paste > 0 else 'off'),
        ('Anchor forcing', f'p {args.teacher_forcing_p0:g} -> 0 over {args.teacher_forcing_epochs} epochs'
         if args.teacher_forcing_p0 > 0 else 'off'),
        ('Loss', f'{len(weights)} weighted terms (focal gamma {args.focal_gamma:g}, missed-detection '
                 f'weight x{args.fn_penalty:g}, hand x{args.hand_fn_penalty:g})'),
    ]
    R.section('Training setup')
    R.kv(setup_table)

    start_epoch = 0
    best_val_loss, patience_counter = float('inf'), 0
    history = defaultdict(list)
    found_zero_fn_epoch, best_tier1_roi_iou, best_tier2_score = False, -1.0, -float('inf')
    best_epoch, best_metric_epoch, best_metric_tier = None, None, None

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        if isinstance(ckpt, dict):
            start_epoch = ckpt.get('epoch', 0)
            # restore full training state; OneCycleLR must resume mid-schedule, not from warmup
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'scaler_state_dict' in ckpt and use_amp:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            for k, v in (ckpt.get('history') or {}).items():
                history[k] = list(v)
            best_val_loss = ckpt.get('best_val_loss', best_val_loss)
            patience_counter = ckpt.get('patience_counter', 0)
            found_zero_fn_epoch = ckpt.get('found_zero_fn_epoch', False)
            best_tier1_roi_iou = ckpt.get('best_tier1_roi_iou', -1.0)
            best_tier2_score = ckpt.get('best_tier2_score', -float('inf'))
            best_epoch = ckpt.get('best_epoch')
            best_metric_epoch, best_metric_tier = ckpt.get('best_metric_epoch'), ckpt.get('best_metric_tier')
        note = '' if 'scheduler_state_dict' in (ckpt if isinstance(ckpt, dict) else {}) else \
            ' [WARN] no scheduler state: LR restarts from warm-up'
        print(f'\n  Resumed from {args.resume} at epoch {start_epoch}.{note}')

    info = {'environment': env, 'dataset': data_summary, 'model': {**profile, 'pretrained_backbone': pretrained},
            'args': vars(args), 'loss_weights': weights,
            'config': {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
            'setup_table': setup_table, 'model_table': model_table,
            'dataset_table': {'headers': ['Split', 'Images'] + [R.SCENE_NAMES[f] for f in R.SCENES],
                              'rows': [[s, d['labels']] + [d['scenes'][R.SCENE_NAMES[f]] for f in R.SCENES]
                                       for s, d in data_summary.items()]}}
    write_run_summary(info, tdir)

    R.section(f'Training (validation metrics; best = lowest val loss, metric = best target IoU with no missed detections)')
    print_epoch_header()
    epoch_times, stop_reason = [], f'reached {args.epochs} epochs'
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        force_p = forcing_prob(epoch, args.teacher_forcing_p0, args.teacher_forcing_epochs)
        lr = optimizer.param_groups[-1]['lr']
        train_loss, tm = train_one_epoch(model, train_dl, criterion, optimizer, scheduler,
                                         scaler, device, args.mixup_alpha, args.grad_clip,
                                         use_amp, force_p, epoch, args.log_every)
        val_loss, vm = validate(model, val_dl, criterion, device, use_amp)
        dt = time.time() - t0
        epoch_times.append(dt)

        history['epoch'].append(epoch + 1)
        history['time_s'].append(dt)
        history['lr'].append(lr)
        history['anchor_forcing_p'].append(force_p)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for k in tm:
            history[f'train_{k}'].append(tm[k])
        for k in vm:
            history[f'val_{k}'].append(vm[k])

        saved = []
        if val_loss < best_val_loss:
            best_val_loss, patience_counter, best_epoch = val_loss, 0, epoch + 1
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss, 'val_metrics': vm, 'args': vars(args)},
                       os.path.join(args.output_dir, 'best_model.pth'))
            saved.append('best')
        else:
            patience_counter += 1

        hand_fn, point_fn, roi_fn = vm['hand_fn_rate'], vm['point_fn_rate'], vm['roi_fn_rate']
        zero_fn = hand_fn == 0.0 and point_fn == 0.0 and roi_fn == 0.0
        metric_path = os.path.join(args.output_dir, 'best_model_by_metric.pth')
        if zero_fn:
            if not found_zero_fn_epoch or vm['roi_iou'] > best_tier1_roi_iou:
                found_zero_fn_epoch, best_tier1_roi_iou = True, vm['roi_iou']
                best_metric_epoch, best_metric_tier = epoch + 1, 1
                torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                            'val_loss': val_loss, 'val_metrics': vm, 'selection_tier': 1,
                            'args': vars(args)}, metric_path)
                saved.append('metric')
        elif not found_zero_fn_epoch:
            score = vm['roi_iou'] - (hand_fn + point_fn + roi_fn)
            if score > best_tier2_score:
                best_tier2_score = score
                best_metric_epoch, best_metric_tier = epoch + 1, 2
                torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                            'val_loss': val_loss, 'val_metrics': vm, 'selection_tier': 2,
                            'selection_score': score, 'args': vars(args)}, metric_path)
                saved.append('metric')

        print_epoch_row(epoch + 1, args.epochs, dt, lr, train_loss, val_loss, vm, '+'.join(saved))

        # crash-safe resume checkpoint every epoch, after the selection bookkeeping above
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'history': dict(history), 'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'found_zero_fn_epoch': found_zero_fn_epoch,
                    'best_tier1_roi_iou': best_tier1_roi_iou,
                    'best_tier2_score': best_tier2_score,
                    'best_epoch': best_epoch, 'best_metric_epoch': best_metric_epoch,
                    'best_metric_tier': best_metric_tier,
                    'val_loss': val_loss, 'val_metrics': vm, 'args': vars(args)},
                   os.path.join(args.output_dir, 'last_checkpoint.pth'))
        save_history(history, tdir)

        if patience_counter >= args.patience:
            stop_reason = f'early stop: no val-loss improvement for {args.patience} epochs'
            break

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    save_history(history, tdir)
    if history['epoch']:
        plot_training_curves(history, best_epoch, tdir)

    total_time = time.time() - run_start
    epochs_run = len(history['epoch'])
    bi = history['epoch'].index(best_epoch) if best_epoch in history['epoch'] else None
    best_epoch_table = [] if bi is None else [
        (label, f'{history[f"val_{k}"][bi]:.4f}') for k, label in TRACKED.items() if f'val_{k}' in history]
    result_table = [
        ('Epochs run', f'{epochs_run} of {args.epochs} ({stop_reason})'),
        ('Training time', f'{R.fmt_duration(total_time)} (median {np.median(epoch_times) if epoch_times else 0:.0f} s/epoch)'),
        ('best_model.pth', f'epoch {best_epoch}, val loss {best_val_loss:.4f}'),
        ('best_model_by_metric.pth', f'epoch {best_metric_epoch}, '
                                     + ('no missed detections on val' if best_metric_tier == 1
                                        else 'target IoU minus miss rate (no epoch reached zero misses)')),
    ]
    R.section('Summary')
    R.kv(result_table)
    if best_epoch_table:
        print(f'\n  Validation metrics at the best epoch ({best_epoch}):')
        R.print_table(['Metric', 'Value'], best_epoch_table, indent=4)

    info.update({'result_table': result_table, 'best_epoch_table': best_epoch_table,
                 'timing': {'total_s': total_time, 'epoch_times_s': epoch_times},
                 'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
                 'best_metric_epoch': best_metric_epoch, 'best_metric_tier': best_metric_tier,
                 'epochs_run': epochs_run, 'stop_reason': stop_reason})
    write_run_summary(info, tdir)
    return model, history, info


def create_quantized_models(args, info):
    """Export FP32/FP16/INT8 copies of best_model.pth and add their sizes to the run summary."""
    best_path = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(best_path):
        print(f'  [WARN] No best_model.pth at {best_path}')
        return
    model = build_model(args)
    state = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    sizes = save_quantized_models(model, args.output_dir)
    limit_kb = args.edge_limit_mb * 1024
    rows = [(f.upper(), f'{d["size_kb"]:.0f}', f'{d["size_kb"] / limit_kb * 100:.0f}% of {args.edge_limit_mb:g} MB')
            for f, d in sizes.items()]
    R.section('Exported models (from best_model.pth)')
    R.print_table(['Format', 'Size (KB)', 'Budget used'], rows)
    info.update({'quantized': sizes, 'quantized_table': rows})
    write_run_summary(info, os.path.join(args.output_dir, 'training'))


def add_model_args(p):
    p.add_argument('--img_size', type=int, default=cfg.IMG_SIZE)
    p.add_argument('--no_bifpn', action='store_true')
    p.add_argument('--no_coordconv', action='store_true')
    p.add_argument('--mixer_depth', type=int, default=cfg.CONV_MIXER_DEPTH)
    p.add_argument('--no_anchor_chain', action='store_true', help='pooled heads')
    p.add_argument('--global_softargmax', action='store_true', help='global soft-argmax expectation')
    p.add_argument('--roi_geometry', choices=['additive', 'multiplicative', 'none'], default='additive')
    p.add_argument('--no_ft_box_prior', action='store_true')
    return p


def parse_args():
    p = argparse.ArgumentParser(description='Train SpatialNet',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_dir', type=str, default=cfg.DATA_DIR)
    p.add_argument('--output_dir', type=str, default=cfg.OUTPUT_DIR)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    p.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    p.add_argument('--lr', type=float, default=cfg.MAX_LR)
    p.add_argument('--patience', type=int, default=cfg.PATIENCE)
    p.add_argument('--grad_clip', type=float, default=cfg.GRAD_CLIP)
    p.add_argument('--mixup_alpha', type=float, default=cfg.MIXUP_ALPHA)
    p.add_argument('--copy_paste', type=float, default=cfg.CP_PROB,
                   help='distractor-hand copy-paste probability (0 disables)')
    p.add_argument('--ownership_weight', type=float, default=cfg.OWNERSHIP_NEG_WEIGHT)
    p.add_argument('--teacher_forcing_p0', type=float, default=cfg.TEACHER_FORCING_P0,
                   help='initial scheduled-anchor-forcing probability (0 disables)')
    p.add_argument('--teacher_forcing_epochs', type=int, default=cfg.TEACHER_FORCING_EPOCHS,
                   help='epochs over which anchor forcing decays to zero')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--log_every', type=int, default=0,
                   help='print in-epoch progress every N batches (0 = one line per epoch)')
    p.add_argument('--load_pretrained', action='store_true', default=cfg.LOAD_PRETRAINED)
    p.add_argument('--backbone_lr_mult', type=float, default=cfg.BACKBONE_LR_MULT)
    add_model_args(p)
    lw = cfg.LOSS_WEIGHTS
    for k in lw:
        p.add_argument(f'--w_{k}', type=float, default=lw[k])
    p.add_argument('--focal_gamma', type=float, default=cfg.FOCAL_GAMMA)
    p.add_argument('--fn_penalty', type=float, default=cfg.FN_PENALTY_MULT)
    p.add_argument('--hand_fn_penalty', type=float, default=cfg.HAND_FN_PENALTY_MULT)
    p.add_argument('--quantize', action='store_true')
    p.add_argument('--edge_limit_mb', type=float, default=cfg.EDGE_MEMORY_LIMIT_MB)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    _, _, run_info = train(args)
    if args.quantize:
        create_quantized_models(args, run_info)
    print(f'\n  Done. Checkpoints in {args.output_dir}, statistics and figures in '
          f'{os.path.join(args.output_dir, "training")}\n')
