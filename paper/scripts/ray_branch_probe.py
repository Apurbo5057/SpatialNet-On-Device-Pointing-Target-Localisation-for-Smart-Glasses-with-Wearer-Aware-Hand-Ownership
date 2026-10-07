"""Does the trained model use its ray-conditioning branch G^t?

Runs the main checkpoint on N test images (CPU) twice: as trained, and with the geometry branch's
output set to zero at inference. Reports the size of G^t relative to the visual logits and the
change in target IoU. Run from the repository root:
    python paper/scripts/ray_branch_probe.py [--n 500] [--model runs/main_run/outputs_spatialnet/best_model.pth]
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import model as M  # noqa: E402
from dataset import get_eval_loader  # noqa: E402
from loss import compute_siou  # noqa: E402
from annotation import fingertip_roi_gap, gap_bin  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.join(ROOT, 'runs', 'main_run', 'outputs_spatialnet', 'best_model.pth'))
ap.add_argument('--n', type=int, default=500)
args = ap.parse_args()
torch.set_num_threads(4)

ck = torch.load(args.model, map_location='cpu', weights_only=False)
m = M.SpatialNet(load_pretrained=False)
m.load_state_dict(ck.get('model_state_dict', ck))
m.eval()

geo_out, vis_out, zero = {}, {}, {'on': False}
m.roi_hm.cond.register_forward_hook(lambda mod, i, o: geo_out.__setitem__('g', o.detach()) or
                                    (torch.zeros_like(o) if zero['on'] else None))
m.roi_hm.visual.register_forward_hook(lambda mod, i, o: vis_out.__setitem__('v', o.detach()))

dl = get_eval_loader(os.path.join(ROOT, 'EgoPoint_spatialnet'), 'test', batch_size=25, num_workers=0)
iou = {'on': [], 'off': []}
ratio, gaps = [], []
seen = 0
with torch.no_grad():
    for b in dl:
        keep = b['roi_conf'].squeeze(1) > 0
        if keep.sum() == 0:
            continue
        for mode in ('on', 'off'):
            zero['on'] = mode == 'off'
            out = m(b['image'])
            _, i = compute_siou(out['roi'][keep], b['roi_bbox'][keep])
            iou[mode] += i.tolist()
            if mode == 'on':
                g, v = geo_out['g'][keep], vis_out['v'][keep]
                ratio += (g.flatten(1).std(1) / v.flatten(1).std(1)).tolist()
        for kp, rb in zip(b['fingertip_kp'][keep].tolist(), b['roi_bbox'][keep].tolist()):
            gaps.append(gap_bin(fingertip_roi_gap(kp, rb)))
        seen += int(keep.sum())
        if seen >= args.n:
            break

on, off, gaps = np.array(iou['on']), np.array(iou['off']), np.array(gaps)
print(f'images: {len(on)}')
print(f'std(G^t) / std(visual logits): median {np.median(ratio):.3f}, p10 {np.percentile(ratio, 10):.3f}, p90 {np.percentile(ratio, 90):.3f}')
for name, msk in [('all', np.ones_like(on, bool)), ('on-object', gaps == 'on-object'), ('near', gaps == 'near'), ('far', gaps == 'far')]:
    print(f'{name:10s} n={msk.sum():4d}  P@0.5 with G^t {100 * (on[msk] >= .5).mean():5.1f}  '
          f'without {100 * (off[msk] >= .5).mean():5.1f}  mIoU {100 * on[msk].mean():5.1f} -> {100 * off[msk].mean():5.1f}')
