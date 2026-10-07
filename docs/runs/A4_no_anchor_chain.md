# Run: ablation A4, no instance-anchored chain (`runs/A4_no_anchor_chain/`)

## Setup

- **Kaggle kernel:** `spatialnet-ablation-a4`.
- **Same as the main run:** code, data, 160 px input and seed 42.
- **Flag:** `--no_anchor_chain`. `run_info.json` confirms it.
  - The direction head and the pointing and target gates read the **whole-image pooled vector**, not the hand and fingertip descriptors.
  - The fingertip heatmap gets no hand-offset branch and no fingertip-inside-hand-box prior.
  - Still present: the ray conditioning of the target heatmap, and the local soft-argmax.
- 709,191 parameters and 0.221 GFLOPs: a plain model of nearly the same size.

## Training health

- No errors in the log. The first download broke on a network error; it was fetched again in full (68 files).
- **It stopped earliest of all runs:** 79 epochs, best checkpoint at epoch 54. The main run went to 125 epochs, best at 100. Its validation loss also levelled off sooner (0.419 vs 0.406). Part of the gap below may come from shorter training.

## What we expected

- Hand, fingertip, direction and target should all belong to the same hand instance, so removing the anchoring should hurt precise localisation and the direction.
- The gates read whole-image features, so the model should be more easily fooled by other people's hands.

## What happened

Paired on the same images; `*` marks a 95% interval that excludes 0.

| Metric | Full (anchored) | A4 (plain) | Difference |
|---|---:|---:|---:|
| Target P@0.5 | 58.9 | 58.6 | −0.3 |
| **Target P@0.7** | 31.8 | 29.0 | **−2.8 *** |
| **Target mIoU** | 53.3 | 51.9 | **−1.4 *** |
| **Small targets (<3% of image), P@0.5** | 36.9 | 25.6 | **−11.3 *** (interval −16.5 to −6.5) |
| Small-target box area, predicted / true | ×1.9 | ×2.4 | worse |
| **Hand mIoU** | 73.7 | 71.9 | **−1.8 *** |
| **Hand P@0.5** | 92.6 | 91.0 | **−1.6 *** |
| **Direction error (degrees)** | 32.2 | 35.6 | **+3.4 *** |
| Far P@0.5 | 25.4 | 25.8 | +0.5 |
| Fingertip error (% of width) | 2.43 | 2.43 | 0.0 |
| Other-hands metrics (P@0.5, no-hand frames kept, ownership) | — | — | no change |

## Interpretation

- **The anchored design matters for precision, not for the headline number.** P@0.5 is unchanged, but:
  - P@0.7 drops by 2.8 points and mIoU by 1.4;
  - small targets drop by 11.3 points, the largest effect of any ablation;
  - the hand box and the direction are clearly worse.
- **Why small targets suffer most:** without anchoring, the target size head and the gates see only coarse whole-image features, so small boxes are drawn even larger (×2.4).
- **The fingertip is unaffected.** Its visual heatmap alone is enough at this level. The hand-offset branch and box prior add no measurable precision on clean images.
- **Not as expected: no extra sensitivity to other hands.** The "instance descriptors protect the gates from other hands" argument is not supported on synthetic distractors. The distractor pasting (A1) does that job. Untested on real other people's hands.
- **Caveat:** A4 trained the shortest (best epoch 54 vs 100), which may inflate the gaps. One seed.

## Verdict

- **Supported:** "The anchored chain is what lets a 0.7M model draw precise boxes and handle small targets." P@0.7 −2.8, small targets −11.3, hand mIoU −1.8, direction +3.4°, all significant.
- **Not supported:**
  - "the anchored chain improves the headline P@0.5";
  - "instance-level gates resist other hands".
- **For the paper:** A4 is the same-size plain baseline. It shows the design matters, but at the precise end of the metric (P@0.7, small targets), not at P@0.5.
