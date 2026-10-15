# SpatialNet training run

## Setup

| Setting | Value |
|---|---:|
| Input size | 160 x 160 |
| Epochs (max) / early-stop patience | 180 / 25 epochs on val loss |
| Batch size / steps per epoch | 32 / 334 |
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | one-cycle, peak 0.003 (backbone x0.1), 10% warm-up, cosine decay |
| Gradient clipping | 1 |
| MixUp | alpha 0.2, within same H P T group |
| Distractor-hand copy-paste | p 0.5 (no-hand p 0.3), up to 2 hands |
| Anchor forcing | p 0.5 -> 0 over 20 epochs |
| Loss | 13 weighted terms (focal gamma 2, missed-detection weight x2, hand x3) |

## Dataset

| Split | Images | No hand | Non-pointing | Pointing, no target | Pointing at target |
|---|---:|---:|---:|---:|---:|
| train | 10669 | 1535 | 0 | 0 | 9134 |
| val | 1524 | 211 | 0 | 0 | 1313 |
| test | 3029 | 425 | 0 | 0 | 2604 |

## Model

| Property | Value |
|---|---:|
| Backbone | ShuffleNetV2, ImageNet-pretrained |
| Parameters | 722,104 (backbone 143,136, heads 578,968) |
| Size (FP32) | 2.75 MB |
| Compute | 0.231 GFLOPs per 160x160 image |

## Result

|  | Value |
|---|---:|
| Epochs run | 107 of 180 (early stop: no val-loss improvement for 25 epochs) |
| Training time | 1h 36m (median 52 s/epoch) |
| best_model.pth | epoch 82, val loss 0.4088 |
| best_model_by_metric.pth | epoch 40, no missed detections on val |

## Validation metrics at the best epoch

| Metric | Value |
|---|---:|
| Hand IoU | 0.6613 |
| Target IoU | 0.4744 |
| Fingertip error (px) | 3.5180 |
| Ray angle error (deg) | 30.6931 |
| Hand gate acc | 0.9870 |
| Pointing gate acc | 0.8958 |
| Target gate acc | 0.8958 |
| Missed hand rate | 0.0052 |
| Missed pointing rate | 0.0000 |
| Missed target rate | 0.0000 |
| Ownership acc | 0.0000 |

## Exported model sizes

| Format | Size (KB) | Budget used |
|---|---:|---:|
| FP32 | 3065 | 30% of 10 MB |
| FP16 | 1601 | 16% of 10 MB |
| INT8 | 2735 | 27% of 10 MB |
