# Runs: seeds 1 and 2 (T4) (`runs/seed_1/`, `runs/seed_2/`)

## Setup

- **Kaggle kernels:** `spatialnet-seed-1` (version 2) and `spatialnet-seed-2`.
- **Same as the main run:** code, data, 160 px input and 180-epoch one-cycle schedule.
- **Only change:** `--seed 1` or `--seed 2`. `run_info.json` confirms the seed; everything else matches the main run.

## Training health

- No errors in either log. The "error" hits are metric names, such as fingertip error and ray error.
- 68 files each, the same as the other runs.

| Run | Epochs run | Best epoch |
|---|---:|---:|
| Seed 42 (main) | 125 | 100 |
| Seed 1 | 128 | 103 |
| Seed 2 | 106 | 81 |

## Results (test, 2,604 images with a target)

| Metric | Seed 42 | Seed 1 | Seed 2 | Mean ± std | Largest change from seed 42 |
|---|---:|---:|---:|---:|---:|
| P@0.3 | 78.6 | 78.2 | 77.5 | 78.1 ± 0.6 | 1.1 |
| **P@0.5** | 58.9 | 59.0 | 59.0 | **59.0 ± 0.1** | 0.1 |
| P@0.7 | 31.8 | 31.7 | 31.0 | 31.5 ± 0.4 | 0.8 |
| mIoU | 53.3 | 53.0 | 52.4 | 52.9 ± 0.5 | 0.9 |
| Far P@0.5 | 25.4 | 27.8 | 25.8 | 26.3 ± 1.3 | 2.4 |
| **Small P@0.5** (309 images) | 36.9 | 35.3 | 27.8 | **33.3 ± 4.8** | **9.1 *** |
| Hand mIoU | 73.7 | 74.0 | 73.4 | 73.7 ± 0.3 | 0.3 |
| Hand P@0.5 | 92.6 | 92.8 | 92.0 | 92.5 ± 0.4 | 0.6 |
| Fingertip error (px) | 3.89 | 3.80 | 3.81 | 3.83 | 0.09 |
| Direction error, mean (°) | 32.2 | 33.5 | 33.0 | 32.9 ± 0.6 | 1.2 * |
| Direction error, median (°) | 18.6 | 20.2 | 19.8 | 19.5 ± 0.8 | 1.6 |
| Direction error, far (°) | 29.0 | 32.3 | 29.6 | 30.3 ± 1.8 | 3.4 |
| P@0.5 with other hands | 60.1 | 59.9 | 60.5 | 60.2 ± 0.3 | 0.4 |
| No-hand frames kept, other hands | 91.8 | 92.7 | 95.1 | 93.2 ± 1.7 | 3.3 * |

`*` marks a paired-bootstrap interval, from seed 42 to that seed, that excludes 0. These are false "effects" caused by retraining alone.

## Interpretation

- **The headline is very stable.** P@0.5 is 58.9–59.0 in all three runs. P@0.7, mIoU and hand vary by under 1 point.
- **Seed 42 is a slightly lucky seed** for P@0.7, mIoU, small targets and direction. It is the deployed model, so it stays the headline, and Table 4 now also reports the mean ± std.
- **Small-target accuracy is noisy:** 27.8–36.9%, and seed 2 is a *significant* 9.1 points below seed 42. With only 309 images, the paired bootstrap alone overstates confidence on this metric.
- **The paired bootstrap does not cover training variation.** Three seed-to-seed differences were "significant" (small targets, direction, no-hand frames kept). This is why each ablation now also has to beat the retraining spread.

## Calibrating the ablations

The rule: a difference counts as an effect only if it is significant **and** larger than the largest seed-to-seed change on that metric.

| Effect | Ablation | Retraining | Verdict |
|---|---:|---:|---|
| A4 P@0.7 | −2.8 | 0.8 | Robust |
| A4 hand mIoU | −1.8 | 0.3 | Robust |
| A4 direction | +3.4° | 1.2° | Robust |
| A4 small targets | −11.3 | 9.1 | Passes, narrowly: supporting evidence only |
| A1 P@0.5 with other hands | −4.1 | 0.4 | Robust |
| A1 no-hand frames kept | −5.4 | 3.3 | Passes |
| A1 clean mIoU, hand, direction | −1.2, −0.6, +1.8° | 0.9, 0.3, 1.2° | Pass, narrowly |
| A2 far direction | +5.6° | 3.4° | Passes |
| A2 overall direction | +1.5° | 1.2° | Narrow: supporting evidence only |
| A3 P@0.7 | +1.9 (ablation better) | 0.8 | Passes: the branch costs a little precision |
| A3 direction | −1.3° (ablation better) | 1.2° | Narrow |
| Any P@0.5 | within ±1.1 | 0.1 | No P@0.5 conclusions |

- **The resolution gains are robust:** P@0.5 +6.2 at 320 px and far P@0.5 +10.5 at 224 px, against 0.1 and 2.4 from retraining.
- **The small-target gain at 320 px** (46.6%) is above all three 160 px seeds (27.8–36.9%), but this metric is noisy.

## Applied to the paper (2026-10-08)

- **Table 4:** filled the "mean ± std, 3 seeds" row (78.1±0.6, 59.0±0.1, 31.5±0.4, 52.9±0.5, 26.3±1.3).
- **Results intro (§7):** a seed-variation summary replaces the placeholder.
- **Table 7:**
  - new "Retraining (seeds 1, 2)" row;
  - bold now means the interval excludes 0 *and* the difference beats retraining (the same cells stay bold).
- **Figure 5:** grey retraining bands; A4 rows reordered to P@0.7, hand, then small.
- **Ablation text:**
  - two-way testing described;
  - A4 bullets give the seed spread;
  - A2 leads with far direction, and the overall 1.5° is marked as narrow;
  - A3 says the branch costs a little precision;
  - the robustness paragraph is rewritten.
- **Resolution (§7.3):** the gains are compared with seed variation, and the small-target claim is qualified.
- **Abstract:** the anchoring bullet now leads with P@0.7 and direction, not small targets; it adds "58.9–59.0% across three seeds".
- **Introduction:** small targets moved to the last bullet, plus "each gain exceeds the variation between training seeds".
