# Run: ablation A2, no geometric consistency losses (`runs/A2_no_geometry_loss/`)

## Setup

- **Kaggle kernel:** `spatialnet-ablation-a2`.
- **Same as the main run:** code, data, 160 px input and seed 42.
- **Flags:** `--w_ray 0 --w_hand_axis 0`. `run_info.json` confirms both weights are 0.0.
  - No ray-alignment loss: the predicted target centre is no longer pulled onto the predicted ray.
  - No hand-axis cone loss.
- The model is unchanged (722,104 parameters).

## Training health

- No errors in the log. The first download broke on a network error; it was fetched again in full (68 files).
- 109 epochs, best checkpoint at epoch 84. The main run went to 125 epochs, best at 100.
- Its validation loss (0.401) is **not comparable** with the main run's, because two loss terms are missing.

## What we expected

- A less consistent pointing direction, especially for far targets, where the ray loss matters most.
- Worse far-target accuracy as a consequence.

## What happened

Paired on the same images; `*` marks a 95% interval that excludes 0.

| Metric | Full | A2 (no consistency losses) | Difference |
|---|---:|---:|---:|
| Target P@0.5 | 58.9 | 60.0 | +1.1 |
| Target P@0.7 | 31.8 | 32.4 | +0.6 |
| Target mIoU | 53.3 | 52.8 | −0.5 |
| Far P@0.5 | 25.4 | 28.7 | +3.3 (interval −2.9 to +9.6) |
| **Direction error (degrees)** | 32.2 | 33.7 | **+1.5 *** |
| **Direction error, far targets (degrees)** | 29.0 | 34.6 | **+5.6 *** |
| Failures with direction >45° | 182 | 222 | +40 |
| Missed pointing frames (%) | 0.58 | 0.27 | −0.31 * |
| Hand, fingertip, other-hands metrics | — | — | no significant change |

## Interpretation

- **Half as expected: the losses do improve the direction**, most of all for far targets (5.6° better with them).
- **Not as expected: the better direction does not reach the target box.** Target accuracy is unchanged, far targets included.
- **This agrees with A3.** The target head relies on a learned proximity prior around the fingertip, not on the predicted ray (`A3_no_roi_geometry.md`). So improving the ray does not move the target.
- The small drop in missed pointing frames (0.58% → 0.27%) is tiny in absolute terms (8 → 7 images out of 2,604) and not meaningful for the paper.

## Verdict

- The consistency losses can be reported as **improving the direction estimate** (supported).
- **Not** as improving target localisation (not supported).
- **Design lesson:** for the ray to help far targets, the target search must be forced to use it, for example with a fixed forward cone prior or by weighting far targets more in training. Learning the prior from scratch with mostly touched targets does not get there.
- **Caveats:** one seed, and a slightly shorter training (best epoch 84 vs 100).
