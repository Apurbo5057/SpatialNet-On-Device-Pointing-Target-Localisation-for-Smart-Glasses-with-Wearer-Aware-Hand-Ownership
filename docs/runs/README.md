# Experiment runs

All runs: same code (`spatialnet-code`), seed 42 (except `seed_1`, `seed_2`), default training recipe (Option A). Downloaded outputs are in `runs/<name>/`.

## How to analyse a run

- Compare against the main run with paired bootstrap intervals: `python paper/scripts/compare_runs.py runs/main_run runs/<name>`.
- Error types and target-size bins: `python paper/scripts/error_breakdown.py runs/<name>`.
- Whether the model uses its ray branch, and what shape the learned prior has: `paper/scripts/ray_branch_probe.py` and `ray_prior_shape.py`.

## Runs

Differences are from the main run, paired on the same test images; `*` marks a significant difference (95% interval excludes 0).

| Run | Change | P@0.5 | P@0.7 | Far P@0.5 | Small P@0.5 | Hand mIoU | Direction (°) | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `main_run` | 160 px, full model | 58.9 | 31.8 | 25.4 | 36.9 | 73.7 | 32.2 | Reference |
| `img224` | 224 px input | **62.4*** | **37.1*** | **35.9*** | 33.0 | **74.9*** | 31.4 | [img224.md](img224.md) |
| `img320` | 320 px, re-converted data | **65.1*** | **40.3*** | **34.9*** | **46.6*** | **76.4*** | 31.6 | [img320.md](img320.md) |
| `A1_no_copypaste` | no distractor pasting | 58.8 | 30.5 | 27.3 | 33.0 | 73.1* | 34.1* | [A1_no_copypaste.md](A1_no_copypaste.md) |
| `A2_no_geometry_loss` | no ray and hand-axis losses | 60.0 | 32.4 | 28.7 | 36.2 | 73.2 | 33.7* | [A2_no_geometry_loss.md](A2_no_geometry_loss.md) |
| `A3_no_roi_geometry` | no ray conditioning | 59.6 | 33.7* | 30.6 | 37.9 | 73.7 | 30.9* | [A3_no_roi_geometry.md](A3_no_roi_geometry.md) |
| `A4_no_anchor_chain` | no anchored heads | 58.6 | 29.0* | 25.8 | 25.6* | 71.9* | 35.6* | [A4_no_anchor_chain.md](A4_no_anchor_chain.md) |
| `seed_1` | main run, `--seed 1` (T4) | 59.0 | 31.7 | 27.8 | 35.3 | 74.0 | 33.5* | [seeds.md](seeds.md) |
| `seed_2` | main run, `--seed 2` (T4) | 59.0 | 31.0 | 25.8 | 27.8* | 73.4 | 33.0 | [seeds.md](seeds.md) |

The ablations matter most in situations the table above does not cover:
- **A1, with other hands in view:** P@0.5 −4.1*, and no-hand frames kept −5.4*.
- **A2, far targets:** direction error +5.6°*.

## Overall findings

1. **P@0.5 is insensitive to every design component.** All four ablations are within ±1.1 points of 58.9, none significant. The design shows up in precision and in specific cases, not in the headline number.
2. **Resolution is by far the biggest lever.** At the same 0.72M parameters:
   - P@0.5 rises by +6.2 points at 320 px;
   - far targets gain +10.5 points at 224 px.
3. **The anchored chain (A4) matters for precision:**
   - small targets −11.3;
   - P@0.7 −2.8;
   - hand mIoU −1.8;
   - direction +3.4° worse.
4. **Distractor pasting (A1) matters for other hands in view** (−4.1 and −5.4), at no clean cost.
5. **The geometry components do not reach the target box:**
   - the consistency losses (A2) improve the direction (far targets: −5.6°) but not the target;
   - the ray conditioning (A3) is not needed;
   - the learned prior is a fingertip-proximity blob without forward direction.
6. **Far targets are limited by direction.** Direction error is about 31–32° in every run except the ablations that make it worse. Far P@0.5 levels off near 35%.

## What this means for the paper's claims (applied to the paper on 2026-10-08)

| Claim | Status | Evidence |
|---|---|---|
| Image-only, single pass, 0.72M parameters, on device | **Holds** | All runs |
| Higher resolution helps at the same parameter count | **Holds, strongly** | img224, img320 |
| Anchored heads let a tiny model localise precisely and handle small targets | **Holds** | A4: P@0.7, small, hand, direction |
| Distractor pasting makes the model robust to other hands | **Holds**, on synthetic distractors | A1 |
| Consistency losses improve the pointing direction | **Holds** | A2: −1.5° overall, −5.6° far |
| Ray conditioning or ray losses improve far-target localisation | **Not supported** | A2, A3; probes show a proximity prior |
| Instance-level gates resist other hands | **Not supported** | A4 shows no change with other hands |
| Small objects are limited by the size head | **Revised** | A4 shows anchoring helps them; 320 px shows image detail helps them |

**Caveats for all ablations:**
- One seed each. The paired intervals capture test-sample variation; the seed runs give the training variation (P@0.5 ±0.1, P@0.7 ±0.8, small targets ±9.1). Every claimed effect beats both; A4 small targets and the overall direction changes only narrowly ([seeds.md](seeds.md)).
- Training lengths differ, because early stopping ended each run at a different epoch:

| Run | Best epoch |
|---|---:|
| Main | 100 |
| A1 | 88 |
| A2 | 84 |
| A3 | 120 |
| A4 | 54 |
| img224, img320 | 59 |
