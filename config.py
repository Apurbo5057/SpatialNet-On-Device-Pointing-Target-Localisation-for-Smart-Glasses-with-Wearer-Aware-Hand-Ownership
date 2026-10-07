"""
Central hyper-parameters for SpatialNet: paths, model, augmentation, training, loss, deployment.
Imported as `cfg` by the other scripts; `python config.py` prints the configuration.
"""
import os

# Paths (override via environment variables)
DATA_DIR = os.environ.get(
    'SPATIALNET_DATA_DIR',
    './EgoPoint_spatialnet')
OUTPUT_DIR = os.environ.get(
    'SPATIALNET_OUTPUT_DIR',
    '/kaggle/working/outputs_spatialnet')
PRETRAINED_WEIGHTS = os.environ.get(
    'SPATIALNET_WEIGHTS',
    './shufflenetv2.pth')

# Model
IMG_SIZE = 160           # input size; 192/224 give a finer grid at ~1.5-2x FLOPs
LOAD_PRETRAINED = True

FPN_OUT_CHANNELS = 192   # BiFPN channel width
HEAD_DIM = 256           # shared feature width after `reduce`
CONV_MIXER_DEPTH = 2
CONV_MIXER_KERNEL = 5

LOCAL_GRID = 3           # neighbourhood sampled around a predicted centre
LOCAL_PROJ_DIM = 128
HAND_ROI_GRID = 5        # RoIAlign-style grid over the predicted hand box
HAND_ROI_EXPAND = 1.15
HAND_DESC_DIM = 128
COND_BOTTLENECK = 48

# Peak-windowed soft-argmax: temperature and window radius (grid cells)
SOFTARGMAX_TEMP = {'hand': 8.0, 'fingertip': 10.0, 'roi': 8.0}
SOFTARGMAX_RADIUS = {'hand': 2, 'fingertip': 1, 'roi': 2}

# Heatmap target Gaussian sigma (grid cells)
HEATMAP_SIGMA = {'hand': 1.5, 'fingertip': 0.8, 'roi': 1.2}

# Fingertip must lie inside the dilated hand box; outside cells get a linear
# logit penalty of STRENGTH per box-width
FT_BOX_PRIOR_DILATION = 1.3
FT_BOX_PRIOR_STRENGTH = 4.0

# Distractor-hand copy-paste augmentation
CP_PROB = 0.5              # paste into hand-present images
CP_ON_NOHAND_PROB = 0.3    # paste into no-hand images (label unchanged)
CP_MAX_DISTRACTORS = 2
CP_SCALE = (0.45, 1.0)     # size relative to the donor hand box
CP_TOP_FRACTION = 0.65     # distractor centre must be in the top 65% of the image
CP_REORIENT_PROB = 0.7     # rotate/flip so the wrist is not at the bottom edge
CP_FEATHER = 0.12          # alpha feather, fraction of crop size
CP_SELF_PASTE_PROB = 0.3   # re-paste the wearer's own hand
CP_DECOY_PROB = 0.15       # paste a non-hand patch
CP_DONOR_POOL = 512        # donor crops pre-extracted at startup to avoid decode cost in epoch 1
CP_MAX_OVERLAP = 0.15      # max fraction of the distractor covered by wearer hand / ROI boxes

# Training
BATCH_SIZE = 32
MAX_LR = 3e-3
EPOCHS = 180
PATIENCE = 25
GRAD_CLIP = 1.0
MIXUP_ALPHA = 0.2
NUM_WORKERS = 4            # JPEG decoding is the bottleneck, not the GPU

# Scheduled anchor forcing: for the first EPOCHS, replace the predicted anchor
# with the ground-truth one on a decaying fraction of samples (P0 = 0 disables)
TEACHER_FORCING_P0 = 0.5
TEACHER_FORCING_EPOCHS = 20

# Loss weights (13 terms). GT comes only from the annotation: H P T flags, hand box,
# fingertip, ROI box (docs/annotation_format.md)
LOSS_WEIGHTS = {
    'hand': 0.10,          # CIoU on hand box                  (GT: hand box)
    'fingertip': 0.18,     # Wing on fingertip                 (GT: fingertip)
    'angle': 0.03,         # cosine on (sin, cos)              (GT: fingertip -> ROI centre; hand centre -> fingertip if T=0)
    'roi': 0.45,           # SIoU on target box                (GT: ROI box)
    'roi_iou_aux': 0.10,   # 1 - IoU on target box             (GT: ROI box)
    'hand_conf': 0.08,     # P(hand present)                   (GT: H)
    'point_conf': 0.10,    # P(pointing | hand)                (GT: P)
    'roi_conf': 0.06,      # P(valid target | pointing)        (GT: T)
    'ray': 0.08,           # target centre on fingertip ray    (no GT, predictions only)
    'hand_axis': 0.02,     # direction within hand-axis cone   (no GT, predictions only)
    'hand_hm': 0.10,       # focal heatmap, hand centre        (GT: hand box centre)
    'ft_hm': 0.10,         # focal heatmap, fingertip          (GT: fingertip)
    'roi_hm': 0.10,        # focal heatmap, target centre      (GT: ROI box centre)
}

# cos of the cone half-angle (60 deg) between hand axis and pointing direction;
# zero penalty inside the cone, linear outside
HAND_AXIS_MARGIN_COS = 0.5

FOCAL_GAMMA = 2.0
FN_PENALTY_MULT = 2.0
HAND_FN_PENALTY_MULT = 3.0

# CenterNet-style heatmap focal parameters
HM_ALPHA = 2.0
HM_BETA = 4.0
OWNERSHIP_NEG_WEIGHT = 4.0  # extra negative weight on distractor-hand centres (0 = off)

# Differential learning rate
BACKBONE_LR_MULT = 0.1
HEAD_LR_MULT = 1.0

# Deployment
EDGE_MEMORY_LIMIT_MB = 10.0
ONNX_OPSET = 17              # grid_sample needs >= 16


def print_config():
    print("=" * 60)
    print("SpatialNet Configuration")
    print("=" * 60)
    print(f"  Data dir:       {DATA_DIR}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Image size:     {IMG_SIZE}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Max LR:         {MAX_LR}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Patience:       {PATIENCE}")
    print(f"  Grad clip:      {GRAD_CLIP}")
    print(f"  MixUp alpha:    {MIXUP_ALPHA}")
    print(f"  Copy-paste:     p={CP_PROB} (no-hand p={CP_ON_NOHAND_PROB}), "
          f"max={CP_MAX_DISTRACTORS}")
    print(f"  Pretrained:     {LOAD_PRETRAINED}")
    print(f"  Focal gamma:    {FOCAL_GAMMA}")
    print(f"  FN penalty:     {FN_PENALTY_MULT}x (point/roi), {HAND_FN_PENALTY_MULT}x (hand)")
    print(f"  Ownership w:    {OWNERSHIP_NEG_WEIGHT}")
    print(f"  Loss weights:   {LOSS_WEIGHTS}")
    print(f"  FPN/head width: {FPN_OUT_CHANNELS} / {HEAD_DIM}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
