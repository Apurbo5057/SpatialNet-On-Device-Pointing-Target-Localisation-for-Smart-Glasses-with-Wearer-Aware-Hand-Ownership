# Training

```bash
python train.py --data_dir <dataset> --output_dir <out> --quantize
```

Settings come from `config.py`; every one can be overridden on the command line. The Kaggle notebook (`notebooks/spatialnet_kaggle.ipynb`) runs exactly this command.

## How the model is trained

| Item | Setting |
|---|---|
| Data | `train/` for learning, `val/` for model selection and early stopping. `test/` is never touched during training. Labels: H P T flags + hand box + fingertip + target box (`annotation_format.md`). |
| Input | RGB, resized to 160 × 160, ImageNet mean/std normalisation |
| Backbone | ShuffleNetV2, ImageNet-pretrained (`shufflenetv2.pth`) |
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | One-cycle: peak 3e-3 for the heads and 3e-4 for the backbone, 10% warm-up, cosine decay |
| Batch / epochs | 32 images; up to 180 epochs |
| Early stopping | Stops after 25 epochs without a lower validation loss |
| Stabilisation | Gradient clipping at 1.0; mixed precision on GPU; seed 42 |
| Loss | 13 weighted terms (`LOSS_WEIGHTS` in `config.py`): box IoU losses for hand and target, Wing loss for the fingertip, cosine loss for the pointing direction, focal losses for the three gates and three heatmaps, and two geometric consistency terms. Missed detections are weighted 2× (3× for the hand gate). |
| Anchor forcing | For the first 20 epochs, some images use the true hand and fingertip positions instead of the predicted ones, starting at 50% and falling to 0. This stabilises early training. |

### Augmentation (training images only, in memory)

The dataset on disk is never modified, and augmentation never adds images. Each epoch shows every training image once, as a new random variant.

| Step | Chance |
|---|---|
| Paste 1–2 hands from other images (labels unchanged: teaches the model to ignore hands that aren't the wearer's) | 50% of hand images, 30% of no-hand images |
| Pasted hand is the wearer's own hand | 30% of pasted hands |
| Paste a random non-hand patch from a no-hand image | 15% chance per image that gets pasted hands; skipped if it would cover the wearer's hand or the target, so about 10% in practice |
| Horizontal flip | 50% |
| Shift ±5% / scale ±10% / rotate ±15° | 40% |
| Perspective warp | 20% |
| Brightness and contrast ±20% | 50% |
| Gaussian blur (kernel 3–5) | 20% |
| MixUp with an image of the same H P T scene (alpha 0.2, labels unmixed) | every batch |

Boxes and the fingertip move with the image. Validation and test images are only resized and normalised.

Pasted hands are the only source of non-wearer hands: EgoPoint images contain the wearer's hand and no one else's. The multi-hand test in `test_evaluate.py --multihand` pastes hands by the same procedure, so it measures robustness to these synthetic distractors, not to real scenes with other people's hands (see `evaluation.md`).

EgoPoint has only two scenes, No hand and Pointing at target (`dataset_egopoint.md`). The pointing and target gates therefore see only positive examples during training.

## Console output

1. **Environment**: GPU or CPU, PyTorch and Python versions, seed, paths.
2. **Dataset**: images per split and per scene (No hand, Non-pointing, Pointing no target, Pointing at target), and target images by fingertip–target distance.
3. **Model**: backbone and whether ImageNet weights loaded, parameter count, FP32 size, GFLOPs per image.
4. **Training setup**: the settings above, as used.
5. **One row per epoch** (validation metrics unless marked *train*):

   | Column | Meaning |
   |---|---|
   | Time, LR | Epoch time; head learning rate at the start of the epoch |
   | Train loss, Val loss | Total weighted loss |
   | Hand IoU, Target IoU | Mean box overlap with the label (0–1) on images that have that box |
   | Tip err px | Mean fingertip distance in pixels at 160 × 160 |
   | Ray err deg | Mean angle between predicted and true pointing direction |
   | Gate acc % H/P/T | Accuracy of the hand, pointing and target gates at threshold 0.5 |
   | Missed % H/P/T | Share of true hands, pointing gestures and targets the gate missed |
   | Saved | `best` = new lowest val loss, `metric` = new best-by-metric checkpoint |

6. **Summary**: epochs run and why it stopped, total time, the epoch behind each checkpoint, and validation metrics at the best epoch.
7. **Exported models** (with `--quantize`): FP32 / FP16 / INT8 sizes against the 10 MB budget.

## Saved files

| File | Contents |
|---|---|
| `best_model.pth` | Weights at the lowest validation loss. **This is the checkpoint evaluated and reported in the paper.** |
| `best_model_by_metric.pth` | Best validation target IoU among epochs with no missed hand, pointing or target. If no epoch reaches that, the best target IoU minus miss rate. |
| `last_checkpoint.pth` | Full training state after every epoch; pass to `--resume` to continue |
| `final_model.pth` | Weights after the last epoch |
| `model_fp32.pth`, `model_fp16.pth`, `model_int8.pth` | Exported copies of `best_model.pth` (`--quantize`). INT8 quantises linear layers only. |
| `training/history.csv` (and `.json`) | One row per epoch, columns below |
| `training/run_info.json` | Environment, dataset counts, model profile, every argument and `config.py` value, loss weights, timings, best epochs, exported sizes |
| `training/summary.md` | Setup, dataset, model, result and validation metrics at the best epoch, as Markdown tables |
| `training/figures/` | PDF + PNG: `curve_loss`, `curve_iou`, `curve_fingertip_error`, `curve_ray_angle_error`, `curve_gate_accuracy`, `curve_learning_rate`, and `training_curves` (all six). A vertical line marks the best epoch. |

### `history.csv` columns

| Column | Meaning |
|---|---|
| `epoch`, `time_s`, `lr`, `anchor_forcing_p` | Epoch number, duration, head learning rate, share of anchors forced to ground truth |
| `train_loss`, `val_loss` | Total weighted loss |
| `*_hand_iou`, `*_roi_iou` | Mean hand / target IoU |
| `*_ft_px_err` | Mean fingertip error, px at 160 × 160 |
| `*_angle_err_deg` | Mean pointing-direction error, degrees |
| `*_hand_conf_acc`, `*_point_conf_acc`, `*_roi_conf_acc` | Gate accuracy at 0.5, each on its own population (all images / hand images / pointing images) |
| `*_hand_fn_rate`, `*_point_fn_rate`, `*_roi_fn_rate` | Missed-positive rate of each gate |
| `*_ray_loss`, `*_axis_loss`, `*_hand_hm` | Geometric consistency losses and the hand heatmap loss |
| `*_ownership_acc` | Share of images with pasted hands where the predicted hand is nearer the wearer's. Training only; validation has no pasted hands, so it reads 0. |

`*` is `train_` or `val_`. These are batch averages computed during training. Report test results from `test_evaluate.py` (`evaluation.md`), not from this file.

## Turning parts off

`--mixup_alpha 0` disables MixUp, `--copy_paste 0` disables pasted hands, and `--teacher_forcing_p0 0` disables anchor forcing. Flip, geometric, colour and blur augmentation always run during training.
