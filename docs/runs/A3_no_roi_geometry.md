# Run: ablation A3, no ray conditioning (`runs/A3_no_roi_geometry/`)

## Setup

- **Kaggle kernel:** `spatialnet-ablation-a3`.
- **Same as the main run:** code, data, 160 px input and seed 42.
- **Flag:** `--roi_geometry none`. The target heatmap gets no ray-coordinate branch $G^t$.
  - 708,999 parameters, 13,105 fewer than the full model.
  - 0.221 instead of 0.231 GFLOPs.

## Training health

- No errors in the log.
- 145 epochs, best checkpoint at epoch 120. The main run went to 125 epochs, best at 100.
- **A3 trained about 20% longer than the main run**, which may favour it slightly.

## What we expected

- Worse far-target accuracy without the ray prior.
- Little change on touched targets.

This was the paper's contribution 2: the ray-conditioned target search.

## What happened

**Not as expected: without the ray branch, the model is not worse.** Paired on the same images; `*` marks a 95% interval that excludes 0.

| Metric | Full (with $G^t$) | A3 (without) | Difference |
|---|---:|---:|---:|
| Target P@0.5 | 58.9 | 59.6 | +0.7 |
| Target P@0.7 | 31.8 | 33.7 | **+1.9 *** |
| Target mIoU | 53.3 | 53.5 | +0.2 |
| **Far P@0.5** | 25.4 | 30.6 | +5.3 (interval −1.0 to +11.0) |
| Near P@0.5 | 49.2 | 50.4 | +1.2 |
| Direction error (degrees) | 32.2 | 30.9 | −1.3 * (better) |
| Hand P@0.5 | 92.6 | 91.7 | −0.9 * |
| Ownership and other-hands metrics | — | — | no significant change |

## Why: probing the trained full model

`paper/scripts/ray_branch_probe.py` and `ray_prior_shape.py`, run on 500–800 test images on the CPU.

- **The full model does use the branch.**
  - Its output is about as large as the visual term (ratio of spreads 0.86).
  - Switching it off at inference drops P@0.5 from 58.8 to 49.8.
- **But the branch is redundant.** Trained without it (A3), the network learns the same thing elsewhere and reaches the same accuracy. "The model relies on it" does not mean "it is needed".
- **The learned prior is a broad blob, not a forward cone.** Correlation of the branch's output with each map, within an image:
  - distance from the fingertip: −0.35;
  - distance from the ray's line: −0.33;
  - **distance along the ray in front of the fingertip: +0.04**, i.e. almost no forward preference;
  - the same for far targets (−0.32 / −0.31 / +0.08), so it does not adapt to distance.
- **It helps only near the finger.** Switching it off at inference:
  - touched targets: P@0.5 −11.0;
  - near targets: −9.5;
  - far targets: no loss (27.2 → 30.4).
- **Likely cause:** about 70% of targets are touched and only 7% are far. The loss is dominated by touched targets, so the zero-initialised branch learned a fingertip-proximity prior that the visual head can also learn by itself.

## Verdict

- **The claim "ray conditioning improves far targets" is not supported. Do not keep it.**
- Possible framings for the paper (to decide later):
  1. Report honestly that the learned ray prior collapses into a proximity prior and is not necessary. The ray coordinates are an input the model can use, but EgoPoint's distance skew gives it little reason to.
  2. Or test a design that forces direction: a fixed (non-learned) forward cone prior, or weighting far targets more in the loss. That would be new code and a new run.
- **Caveats:** one seed, and A3 trained 20% longer. The +1.9 on P@0.7 may partly be training length, but there is clearly no loss from removing the branch.
- **The paper's §4.3, §7.4 (A3 row), abstract and contributions must be revised.** Not done yet, as requested.
