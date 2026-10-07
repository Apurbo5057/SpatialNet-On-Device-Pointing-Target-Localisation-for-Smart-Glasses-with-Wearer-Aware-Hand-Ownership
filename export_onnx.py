"""Export SpatialNet to ONNX and verify ONNX Runtime parity against PyTorch.

    python export_onnx.py --model_path <out>/best_model.pth --out <out>/spatialnet.onnx [--verify_image <image>]
Input: normalized RGB float32 [1, 3, H, W]. Outputs: hand, fingertip, angle, roi,
hand_conf, point_conf, roi_conf (gates are raw logits).
"""

import os
import argparse
import tempfile
import warnings
import numpy as np
import torch

from model import SpatialNet, ExportWrapper
import config as cfg
from train import add_model_args

# shape-dependent Python branches are constant for a fixed input size; parity is verified below
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings('ignore', message='.*legacy TorchScript-based ONNX export.*')


def load_model(path, args):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    saved = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    model = SpatialNet(
        input_size=saved.get('img_size', args.img_size), load_pretrained=False,
        use_bifpn=not saved.get('no_bifpn', args.no_bifpn),
        use_coordconv=not saved.get('no_coordconv', args.no_coordconv),
        mixer_depth=saved.get('mixer_depth', args.mixer_depth),
        anchor_chain=not saved.get('no_anchor_chain', args.no_anchor_chain),
        local_softargmax=not saved.get('global_softargmax', args.global_softargmax),
        roi_geometry=saved.get('roi_geometry', args.roi_geometry),
        ft_box_prior=not saved.get('no_ft_box_prior', args.no_ft_box_prior))
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    return model.eval()


def export(model, out_path, img_size, opset=cfg.ONNX_OPSET):
    wrapper = ExportWrapper(model).eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    names = ['hand', 'fingertip', 'angle', 'roi', 'hand_conf', 'point_conf', 'roi_conf']
    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, out_path, opset_version=opset,
                          input_names=['image'], output_names=names,
                          do_constant_folding=True, dynamo=False)
    size_mb = os.path.getsize(out_path) / 1024 ** 2
    print(f'  exported {out_path} ({size_mb:.2f} MB, opset {opset})')
    return names


def verify(model, onnx_path, img_size, n=3, image_path=None, tolerance=1e-4):
    import onnx
    import onnxruntime as ort

    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    names = ['hand', 'fingertip', 'angle', 'roi', 'hand_conf', 'point_conf', 'roi_conf']
    if [item.name for item in sess.get_inputs()] != ['image']:
        raise RuntimeError('ONNX input name differs from the export contract')
    if [item.name for item in sess.get_outputs()] != names:
        raise RuntimeError('ONNX output names or order differ from the export contract')
    wrapper = ExportWrapper(model).eval()
    worst = 0.0
    generator = torch.Generator().manual_seed(0)
    inputs = [torch.randn(1, 3, img_size, img_size, generator=generator) for _ in range(n)]
    if image_path:
        from PIL import Image
        from inference import preprocess
        with Image.open(image_path) as image:
            real_input, _ = preprocess(image.convert('RGB'), img_size)
        inputs.append(real_input)
    for x in inputs:
        with torch.no_grad():
            ref = [t.numpy() for t in wrapper(x)]
        out = sess.run(names, {'image': x.numpy()})
        for name, r, o in zip(names, ref, out):
            if r.shape != o.shape or not np.isfinite(o).all():
                raise RuntimeError(f'Invalid ONNX output {name}: expected {r.shape}, got {o.shape}')
            worst = max(worst, float(np.abs(r - o).max()))
    print(f'  ONNX graph valid; max abs diff over {n} random input(s)'
          f'{" + 1 real image" if image_path else ""}: {worst:.2e}')
    if worst > tolerance:
        raise RuntimeError(f'ONNX parity failed: {worst:.2e} > {tolerance:.2e}')
    return worst


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, default=os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
    p.add_argument('--out', type=str, default='spatialnet.onnx')
    p.add_argument('--random_init', action='store_true', help='export an untrained model (smoke test)')
    p.add_argument('--verify_image', type=str, default=None,
                   help='real image for ONNX Runtime parity (uses inference preprocessing)')
    p.add_argument('--tolerance', type=float, default=1e-4,
                   help='maximum allowed absolute difference from PyTorch')
    add_model_args(p)
    args = p.parse_args()

    if args.random_init:
        model = SpatialNet(
            input_size=args.img_size, load_pretrained=False,
            use_bifpn=not args.no_bifpn,
            use_coordconv=not args.no_coordconv,
            mixer_depth=args.mixer_depth,
            anchor_chain=not args.no_anchor_chain,
            local_softargmax=not args.global_softargmax,
            roi_geometry=args.roi_geometry,
            ft_box_prior=not args.no_ft_box_prior).eval()
    else:
        model = load_model(args.model_path, args)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix='.onnx_export_', suffix='.onnx', dir=out_dir)
    os.close(fd)
    try:
        export(model, temp_path, model.input_size)
        verify(model, temp_path, model.input_size,
               image_path=args.verify_image, tolerance=args.tolerance)
        os.replace(temp_path, args.out)
        print(f'  saved verified model to {args.out}')
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
