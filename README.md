# SpatialNet: On-Device Pointing-Target Localisation for Smart Glasses with Wearer-Aware Hand Ownership

SpatialNet is a PyTorch research implementation for locating the target of a pointing gesture in egocentric images. It predicts the wearer's hand box, fingertip, pointing direction, target box, and confidence gates for hand presence, pointing, and a valid target. Wearer-aware training uses distractor-hand copy-paste augmentation and ownership-weighted heatmap supervision.

The model combines a ShuffleNetV2 backbone, BiFPN features, coordinate-aware processing, and chained instance-anchored prediction heads. The default input is 160 x 160 pixels. ONNX export and optional quantization support deployment experiments; actual smart-glasses performance must be measured on the intended hardware.

## Repository contents

| Path | Purpose |
| --- | --- |
| `model.py` | SpatialNet architecture and export wrapper |
| `config.py` | Paths, architecture, augmentation, loss, and training defaults |
| `annotation.py`, `dataset.py` | Annotation parsing, data loading, and augmentation |
| `loss.py`, `train.py` | Multi-task losses and training |
| `test_evaluate.py`, `reporting.py` | Evaluation, metrics, figures, and reports |
| `inference.py` | Image-folder, video, and camera inference |
| `export_onnx.py`, `utils.py` | ONNX export, parity checks, and quantization utilities |
| `smoke_test.py` | Synthetic end-to-end pipeline check |
| `egopoint_preprocessing/prepare_egopoint.py` | EgoPoint conversion and dataset reports |
| `notebooks/spatialnet_kaggle.ipynb` | Kaggle workflow |
| `shufflenetv2.pth` | Included backbone initialization weights; not a trained SpatialNet checkpoint |

Datasets are local and excluded from version control. Paper and documentation folders are being developed separately and are omitted from the initial commit; their folders are not ignored.

## Setup

Use a Python environment with a compatible PyTorch installation. For GPU training, install the PyTorch build appropriate for your CUDA environment. From the repository root, a basic PowerShell setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy pillow matplotlib albumentations opencv-python onnx onnxruntime
```

Dependencies are not currently version-pinned. Albumentations enables the augmentation pipeline; OpenCV is used for video/camera inference, and ONNX packages are used for export verification.

## Prepare the dataset

Obtain EgoPoint separately and place its source `img/` and `json/` folders beneath `EgoPoint_dataset/dataset/`. Convert it with:

```powershell
python egopoint_preprocessing/prepare_egopoint.py --src EgoPoint_dataset/dataset --out EgoPoint_spatialnet
```

The loader expects this layout:

```text
EgoPoint_spatialnet/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

Each image has a matching text label containing 13 numbers:

```text
H P T
hand_cx hand_cy hand_width hand_height
fingertip_x fingertip_y
target_cx target_cy target_width target_height
```

Coordinates are normalized to [0, 1]. The valid flags are `0 0 0` (no wearer hand), `1 0 0` (non-pointing hand), `1 1 0` (pointing without a valid target), and `1 1 1` (pointing at a valid target). Absent geometry is written as zeros; boxes use center, width, and height.

## Train and evaluate

The default output path in `config.py` targets Kaggle. Specify a local output directory when running on your computer:

```powershell
python train.py --data_dir EgoPoint_spatialnet --output_dir outputs --num_workers 0
python test_evaluate.py --model_path outputs/best_model.pth --data_dir EgoPoint_spatialnet --output_dir outputs/evaluation
```

Training saves checkpoints, history, statistics, and figures. Resume a run with:

```powershell
python train.py --data_dir EgoPoint_spatialnet --output_dir outputs --resume outputs/last_checkpoint.pth --num_workers 0
```

Use `--multihand` during evaluation for the synthetic distractor-hand evaluation. Configuration paths can also be set through `SPATIALNET_DATA_DIR`, `SPATIALNET_OUTPUT_DIR`, and `SPATIALNET_WEIGHTS`. Run individual scripts with `--help` for training settings and architecture ablations.

## Inference and export

Use a trained SpatialNet checkpoint:

```powershell
python inference.py --model_path outputs/best_model.pth --image_dir path/to/images --out_dir outputs/inference --heatmaps
python inference.py --model_path outputs/best_model.pth --video path/to/video.mp4 --out_dir outputs/video
python inference.py --model_path outputs/best_model.pth --video camera --out_dir outputs/camera
python export_onnx.py --model_path outputs/best_model.pth --out outputs/spatialnet.onnx
```

ONNX export uses opset 17 and checks parity with PyTorch. The input is normalized RGB float32 `[1, 3, H, W]`; outputs are `hand`, `fingertip`, `angle`, `roi`, `hand_conf`, `point_conf`, and `roi_conf`. Confidence outputs are raw logits. Video mode applies gate hysteresis and temporal smoothing. Add `--quantize` to training to request the quantized artifacts supported by the utilities.

## Pipeline checks

```powershell
python model.py
python smoke_test.py --epochs 2
```

The model self-test checks shapes and gradients. The smoke test creates synthetic data and exercises training, evaluation, export, and inference. These checks validate the pipeline; they do not establish accuracy on real-world images.

## Working with Git

Raw/prepared datasets, Python caches, local environments, credentials, and generated outputs are ignored. Keep credentials such as `kaggle.json` local. Add selected paper/documentation files explicitly when they are ready, and inspect `git diff --cached` before committing.
