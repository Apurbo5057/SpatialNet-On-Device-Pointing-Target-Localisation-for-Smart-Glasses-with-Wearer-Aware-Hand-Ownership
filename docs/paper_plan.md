# IMWUT paper plan: SpatialNet

No plan guarantees acceptance. This one covers what IMWUT reviewers check, with the least extra work.

## 1. Positioning

**Title idea:** *SpatialNet: On-Device Pointing-Target Localisation for Smart Glasses with Wearer-Aware Hand Ownership*

**Contributions, in priority order:**

| # | Contribution | Why it is novel |
|---|---|---|
| C1 | **Pointing-target localisation from one egocentric frame**: fingertip → ray → target box, with geometric conditioning, a ray-alignment loss and the anchor chain | The core novelty. It finds *what* is pointed at, not just the hand |
| C2 | **Wearer-aware hand ownership**: ignores other people's hands (distractor copy-paste + ownership penalty) | Bystander hands are a real problem for glasses |
| C3 | **A complete on-device pipeline**: 722K parameters, 0.23 GFLOPs, 1.6 MB (FP16), running on Samsung WearOS glasses | Contrast with VLMs: cloud, seconds, gigabytes |
| C4 | **Evaluation protocol**: fingertip–target distance strata, ownership test, confidence intervals | Makes far-target pointing measurable |

**Phrase "first" carefully:** say "to our knowledge, the first single-stage, on-device model that…" and back it with the related-work table.

**Do not claim:** 4-scene recognition, pointing or target gate accuracy (EgoPoint has no negatives for them), robustness to real bystanders unless task T5 is done.

## 2. Section by section

| Section | Must contain | Data source | Highlight |
|---|---|---|---|
| Abstract | Problem → method → 4 numbers | `evaluation/summary.md`, device table | Target IoU@0.5, far-target result, on-glasses latency, vs-VLM gap |
| 1 Introduction | Pointing on glasses ("what is that?"), VLMs too slow and cloud-bound, the bystander-hand problem, C1–C4, teaser figure | Qualitative figure | One sentence each for C1–C3, and the latency gap to VLMs |
| 2 Related work | Egocentric hands and pointing, referring and pointing-target datasets (EgoPoint), VLM grounding, efficient detectors | Literature | **Comparison table**: method × target output × on-device × wearer-aware × latency |
| 3 Method | Architecture figure; H P T output; heads; **target head in detail** (ray, conditioning, anchor chain); losses; copy-paste with example figure | `model.py`, `docs/training.md` | Spend the most space on the target branch (C1) |
| 4 Dataset | EgoPoint, conversion rules, 232 dropped records and why, statistics, label examples | `docs/dataset_egopoint.md`, figures | Honesty about the drops (reviewers like it) |
| 5 Implementation | Hyper-parameters, T4 GPU, training time, seeds | `training/run_info.json` | — |
| 6 Experiments | 6.1 Metrics · 6.2 **Target localisation vs VLMs** · 6.3 Distance strata · 6.4 Ablations · 6.5 Wearer ownership · 6.6 On-device efficiency · 6.7 Qualitative and failures | `evaluation/`, VLM results, ablation runs, device logs | Lead with 6.2 and 6.3 (C1) |
| 7 User study (optional, see T6) | Task success, time, SUS, interviews | Study logs | Real use on the glasses |
| 8 Discussion | Failure modes (from `predictions.csv`), design implications, limitations | Failure analysis | Turn the limitations into future work |
| 9 Ethics and privacy | Bystanders, on-device = no frames leave the glasses, data license | Text | On-device is a privacy advantage |

**Figures:** teaser · architecture · copy-paste examples · IoU success curve · target IoU by distance · ownership examples · on-glasses photo or screenshot · qualitative successes and failures.

**Tables:** related-work comparison · dataset · main results vs VLMs · ablations · ownership · efficiency on GPU and on glasses.

## 3. Claims and how to defend them

| Claim | Evidence | Likely reviewer objection → answer |
|---|---|---|
| Accurate target localisation | Target IoU mean with CI, ≥0.3/0.5/0.7, by distance | "Mean IoU is low" → report the success rates; far targets are hard for every method, including VLMs (show it) |
| Better than VLMs for this task | Same test images, same metrics | "Unfair, VLMs aren't tuned" → VLMs are the only general alternative; document prompts and versions; add latency and cost |
| Geometry matters | Ablations A2 and A3 (below) | "Gains are noise" → seeds (T4) or bootstrap CIs |
| Wearer-aware | Ownership accuracy, synthetic + real frames (T5) | "Synthetic distractors only" → real-frame check (T5) |
| Runs on glasses | Device table (T2) | "Desktop numbers" → measured on the glasses |
| Practical | User study (T6) or deployment observations | "No users" → small study, or state it as future work |

## 4. Remaining tasks: minimal versions

### T1 Baselines (VLMs, already benchmarked): make the comparison valid

- [ ] Re-run on **the same EgoPoint test split** (3,029 images, or a fixed random 500 with CIs if cost is a problem).
- [ ] Convert VLM output to a box → the same metrics (target IoU mean, ≥0.3/0.5/0.7). If a VLM returns only a point: "point inside the target box" accuracy, for both methods.
- [ ] Record: model name and version, date, prompt (in the appendix), output parsing, failure rate (unparseable answers).
- [ ] Add **latency per image** and **cost or size** for each VLM, next to SpatialNet's.
- [ ] Report by fingertip–target distance as well (on-object / near / far).

### T2 On-device (Samsung WearOS glasses): minimal data

| Measure | How |
|---|---|
| Device | SoC, RAM, OS version; call it a developer test unit |
| Runtime and format | e.g. ONNX Runtime or TFLite, CPU/GPU/NNAPI delegate, FP32 or FP16 |
| Model latency | Median and p90 over ≥200 frames after 20 warm-up frames (model call only) |
| End-to-end latency | Camera frame → result shown; median over ≥100 frames |
| Throughput | Sustained FPS over 60 s |
| Memory | Peak app memory (Android Studio profiler or `dumpsys meminfo`) |
| Model file size | The deployed file |
| Accuracy parity | Run ~100 test images on the glasses and compare with PyTorch outputs (mean box difference, or same-scene agreement %) |
| *Optional* energy | Battery % drop over 10 min of continuous inference, or the BatteryManager current average |

One table plus one photo of the glasses running it is enough.

### T3 Ablations: 3 runs are necessary and sufficient

| Run | Flags added to `train.py` | Tests |
|---|---|---|
| A1 No distractor copy-paste | `--copy_paste 0 --ownership_weight 0` | C2: ownership drop on the multi-hand test |
| A2 No geometric losses | `--w_ray 0 --w_hand_axis 0` | C1: target IoU, especially far targets |
| A3 No target conditioning | `--roi_geometry none` | C1: the target branch design |

- Same seed and settings as the main run; evaluation runs automatically in the same notebook run.
- Report: target IoU (mean, ≥0.5), far-gap target IoU, ownership accuracy, hand IoU.
- Optional fourth run if compute allows: `--no_anchor_chain`.
- **Loss weights:** no weight search is needed. A2 covers the novel loss terms; the other 10 are standard (CIoU, Wing, focal), so cite them. Put all 13 weights in an appendix table, with one true sentence on how they were set (validation only). Optional sensitivity check: `--w_ray 0.04` and `--w_ray 0.16` (2 runs).

#### How to run each extra run (ablations, seeds, sensitivity)

Every extra run is one new version of the `spatialnet-train` notebook on Kaggle, with two small edits. Training and evaluation both run inside it.

| Run | `OUTPUT_DIR` name (cell 1) | Extra flags (cell 3) |
|---|---|---|
| Main (already running) | `outputs_spatialnet` | none |
| A1 No copy-paste | `ablation_A1_no_copypaste` | `'--copy_paste', '0', '--ownership_weight', '0'` |
| A2 No geometric losses | `ablation_A2_no_geometry_loss` | `'--w_ray', '0', '--w_hand_axis', '0'` |
| A3 No target conditioning | `ablation_A3_no_roi_geometry` | `'--roi_geometry', 'none'` |
| Seed 1 | `seed_1` | `'--seed', '1'` |
| Seed 2 | `seed_2` | `'--seed', '2'` |
| *Optional* w_ray ×0.5 | `sens_wray_0.04` | `'--w_ray', '0.04'` |
| *Optional* w_ray ×2 | `sens_wray_0.16` | `'--w_ray', '0.16'` |

**Steps** (one run at a time; wait until the main run has finished):

1. On Kaggle, open `spatialnet-train` → **Edit**.
2. **Cell 1**: keep `QUICK_RUN = False`, and change the output folder:

   ```python
   OUTPUT_DIR = '/kaggle/working/ablation_A1_no_copypaste'   # name from the table
   ```

3. **Cell 3**: add the run's flags to the train command:

   ```python
   cmd = [sys.executable, 'train.py', '--data_dir', DATA_DIR, '--output_dir', OUTPUT_DIR, '--quantize',
          '--copy_paste', '0', '--ownership_weight', '0']   # flags from the table
   ```

4. **Save Version → Save & Run All (Commit)**, with the GPU accelerator on.
5. When it finishes: open the version → **Output** → download the folder. Do this before the next run, because each new version shows only its own output.
6. Undo both edits (or set them for the next run) and repeat.

**Checks:**

- The training log's "Training setup" block shows the change: copy-paste `off`, or 13 loss terms with the changed weight.
- Ablation checkpoints evaluate correctly automatically, because the architecture flags are stored in each checkpoint.

**Building the ablation table:** put each downloaded run folder side by side (e.g. `runs/outputs_spatialnet`, `runs/ablation_A1_no_copypaste`, …), then run:

```python
import json, os
rows = []
for run in sorted(os.listdir('runs')):
    m = json.load(open(f'runs/{run}/evaluation/metrics.json'))
    loc = m['test']['localisation']
    own = m.get('test_multihand', {}).get('ownership', {})
    rows.append([run,
                 100 * loc['target_iou']['All']['mean'], 100 * loc['target_iou']['All']['@0.5'],
                 100 * loc['target_iou']['Gap far'].get('@0.5', float('nan')),
                 100 * loc['hand_iou']['All hand-present']['mean'],
                 loc['fingertip_px']['All pointing']['mean'],
                 100 * (own.get('ownership_accuracy') or float('nan'))])
print('run | target IoU | target IoU>=0.5 | far target IoU>=0.5 | hand IoU | tip err px | ownership %')
for r in rows:
    print(f'{r[0]} | ' + ' | '.join(f'{v:.1f}' for v in r[1:]))
```

For seeds: average the main run, `seed_1` and `seed_2` for each column (mean ± std).

**Compute:** each run is one full training (several hours on a T4). Kaggle gives about 30 GPU-hours per week, so 5 extra runs take about 1–2 weeks. Start A1–A3 first; seeds next; sensitivity only if time remains.

### T4 Seeds: minimal

- [ ] Train the **full model** 2 more times (`--seed 1`, `--seed 2`; runbook in T3) → report **mean ± std over 3 seeds** for the main table.
- [ ] Ablations: 1 seed each, stated in the paper.
- If compute runs out: keep 1 seed, report the bootstrap CIs already produced, and list it as a limitation (weaker, but common).

**Compute:** Kaggle gives about 30 GPU-hours per week. Main run + 2 seeds + 3 ablations = 6 runs, so spread them over 2 weeks.

### T5 Real multi-person check: the cheap version

You cannot fully skip it, but you can keep it small:

1. Record 3–5 short clips with the glasses (or phone at eye level) where **another person's hand is visible**, and sometimes pointing too. Get consent; avoid faces.
2. Sample **100–150 frames**.
3. Run the model and save each frame with the predicted hand box drawn.
4. Two people label each frame by eye:
   - "predicted hand = wearer's hand" (correct);
   - "predicted hand on the other person's hand" (ownership error);
   - "missed".
5. Report: ownership accuracy with a 95% CI (Wilson), the agreement between the two labellers, and a figure of 6 examples.

No box labelling is needed. About one day of work. This turns "synthetic only" into "synthetic + real check".

### T6 User study: what it is and the minimum

**What:** real people use the system on the glasses for real tasks, and you measure how well it works for them.

**Minimum viable study:**

| Item | Choice |
|---|---|
| Participants | 12 (common IMWUT minimum); record age, gender, handedness, glasses/AR experience |
| Ethics | Institutional approval or exemption, consent form; no recording of bystanders' faces |
| Setup | A room with 10–15 everyday objects at near (<1 m) and far (2–4 m) distances |
| Task | The experimenter names an object → the participant points → the glasses highlight or speak the detected target |
| Trials | 20 per participant (10 near, 10 far) |
| Conditions | SpatialNet on glasses. Optional but strong: the same task with a VLM baseline (cloud) |
| Logged per trial | Target object, predicted object, correct (yes/no), time from pointing to result |
| After the session | SUS (10 items), NASA-TLX raw (6 items), 5-question interview |
| Session length | About 30 min per person |

**Report:**
- task success rate with CI, near vs far;
- median response time;
- SUS score against the benchmark of 68;
- TLX scores;
- with 2 conditions, a paired Wilcoxon test;
- 3–4 interview themes with quotes.

**Effort:** about 1 week (prototype ready + 12 sessions + analysis).

**If there's no time:** IMWUT does accept strong systems papers without a user study, if the on-device deployment is real (T2) and the evaluation is thorough. Then write a "deployment observations" paragraph (what worked and what failed while using it) and put the study under future work. **This is the riskiest gap; do it if at all possible.**

## 5. Priority order (if time is short, stop from the bottom)

1. Main training run (running) → main results.
2. **T1** VLM comparison on the same split.
3. **T2** on-glasses table.
4. **T3** three ablations.
5. **T5** real multi-person check (1 day).
6. **T4** extra seeds.
7. **T6** user study.

## 6. Three-and-a-half-week schedule (November 1)

| Week | Do | Write |
|---|---|---|
| 1 | Main run finishes; start A1–A3 on Kaggle; re-run VLMs on the EgoPoint test split | Method, Dataset, Implementation |
| 2 | Seeds 1 and 2; on-glasses measurements (T2); real multi-person clips (T5) | Related work, Experiments 6.1–6.3 |
| 3 | User study sessions (T6) if possible; failure analysis from `predictions.csv` | Experiments 6.4–6.7, Discussion, Ethics |
| 4 (partial) | Figures and tables final | Abstract, Introduction, proofreading, anonymisation |

**Before submitting:**
- [ ] Anonymise the paper and code links.
- [ ] Check that EgoPoint's license allows sharing the converted data.
- [ ] Make sure every number in the paper comes from `metrics.json`.
