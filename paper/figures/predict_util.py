"""Run the paper checkpoint on single images for figures (CPU, a handful of images only)."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import model as M  # noqa: E402
from dataset import KeypointPointingDataset  # noqa: E402

CKPT = os.path.join(ROOT, 'runs', 'main_run', 'outputs_spatialnet', 'best_model.pth')
DATA = os.path.join(ROOT, 'EgoPoint_spatialnet')

_model, _tf = None, None


def predict(img):
    """img: HxWx3 uint8 RGB. Returns normalised hand box, fingertip, unit direction (dx, dy), target box, probs."""
    global _model, _tf
    if _model is None:
        ck = torch.load(CKPT, map_location='cpu', weights_only=False)
        _model = M.SpatialNet(load_pretrained=False)
        _model.load_state_dict(ck.get('model_state_dict', ck))
        _model.eval()
        ds = KeypointPointingDataset(DATA, 'val', augment=False)
        # albumentations resize when installed (as in evaluation), otherwise the dataset's PIL fallback
        _tf = (lambda im: ds.eval_transform(image=im)['image']) if ds.eval_transform else ds._np_to_tensor
    with torch.no_grad():
        o = _model(_tf(img).unsqueeze(0), return_all=False)
    sin, cos = o['angle'][0].tolist()
    return {'hand': o['hand'][0].numpy(), 'tip': o['fingertip'][0].numpy(), 'dir': np.array([cos, sin]),
            'roi': o['roi'][0].numpy(), 'p_hand': torch.sigmoid(o['hand_conf'])[0, 0].item()}
