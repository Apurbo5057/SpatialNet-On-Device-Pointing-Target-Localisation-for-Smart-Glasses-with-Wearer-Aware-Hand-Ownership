# Run: ablation A1, no distractor-hand pasting (`runs/A1_no_copypaste/`)

## Setup

- **Kaggle kernel:** `spatialnet-ablation-a1`.
- **Same as the main run:** code, data, 160 px input and seed 42.
- **Flags:** `--copy_paste 0 --ownership_weight 0`.
  - No hands from other images are pasted during training.
  - No extra penalty on hand-heatmap responses at pasted hands.
- The model itself is unchanged (722,104 parameters).

## Training health

- No errors in the log; the training summary confirms "copy-paste: off".
- Early stopping after 113 epochs; best checkpoint at epoch 88. The main run went to 125 epochs, best at 100. The two runs therefore trained for slightly different lengths.

## What we expected

- Worse ownership when other hands are visible: more predictions on the other hand.
- More no-hand frames wrongly read as containing a hand, when someone else's hand is present.
- About the same accuracy on clean images, since the clean test images contain no other hands.

## What happened

Paired on the same test images; `*` marks a 95% bootstrap interval that excludes 0.

| Metric | Main (with pasting) | A1 (without) | Difference |
|---|---:|---:|---:|
| **No-hand frames kept with pasted hands (%)** | 91.8 | **86.4** | **−5.4 *** |
| **Target P@0.5 with pasted hands** | 60.1 | **56.1** | **−4.1 *** |
| Predicted centre on a pasted hand (%) | 0.21 | 1.22 | +1.0 * (6× more) |
| Ownership accuracy, pasted hands (%) | 99.7 | 98.7 | −1.0 * |
| Target P@0.5, clean images | 58.9 | 58.8 | −0.1 |
| Target mIoU, clean images | 53.3 | 52.1 | −1.2 * |
| Target P@0.3, clean images | 78.6 | 76.9 | −1.7 * |
| Hand mIoU, clean images | 73.7 | 73.1 | −0.6 * |
| Direction error (degrees) | 32.2 | 34.1 | +1.8 * |
| No-hand false alarms, clean (%) | 6.6 | 6.6 | 0.0 |

## Interpretation

- **As expected: pasting buys robustness to other hands at no cost on clean images.**
  - With other hands in view, target P@0.5 drops by 4.1 points without it.
  - 5.4 points more no-hand frames are misread as "hand present".
  - Clean P@0.5 is unchanged (−0.1).
- **Smaller than expected: the ownership accuracy metric barely moves** (99.7 → 98.7).
  - Even without pasting, the model rarely chooses the pasted hand. EgoPoint's wearer hands come from the bottom of the frame, and pasted hands are placed in the upper part, so position alone separates them.
  - This confirms what the paper says: that metric is near-trivial, and the meaningful measures are no-hand frames kept and target accuracy under distractors.
- **Unexpected bonus: pasting also helps slightly on clean images.** mIoU +1.2, P@0.3 +1.7, direction error −1.8°, hand mIoU +0.6. It probably acts as a regulariser, forcing the model to use the hand's position and context rather than its appearance alone.

## Caveats

- **The distractor test favours the full model.** Its pasted hands come from the same procedure as training, so they are familiar to the full model and new to A1. The real-frame check (T5) is still needed.
- **One seed per run, and slightly different training lengths** (best epoch 88 vs 100). Differences of about 1 point on clean images may be training noise.

## Use in the paper

- Table 6, row A1.
- Table 7, row "A1, pasted hands".
- §7.3 and §7.4: write the claim as "pasting makes the model robust to other hands (−4.1 and −5.4 points without it) at no clean-accuracy cost". Do not cite the 99.7% ownership accuracy as evidence.
