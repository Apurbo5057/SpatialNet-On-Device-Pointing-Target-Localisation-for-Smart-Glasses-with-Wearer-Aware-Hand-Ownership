# IMWUT paper plan: SpatialNet

Status on 2026-10-07.

- Draft: `paper/main.pdf`.
- Map of every red placeholder to its source: `paper/README.md`.
- Each task below lists its goal, the steps, what to record, where it goes in the paper, and when it is done.

---

## 0. Where things stand

### Claim of the paper

- Input: one RGB image only (no text, no hand box, no depth, no external pose model).
- Model: a single forward pass of a 0.72M-parameter CNN without self-attention (0.23 GFLOPs, 1.6 MB in FP16).
- Runs on smart-glasses hardware.
- Accuracy is reported as a trade-off against cost, never as superiority.
- Wearer-ownership is a secondary contribution.

### Done

- Main run at 160 px, seed 42, `runs/main_run/`, checkpoint `best_model.pth`. Its numbers are already in the paper:
  - target P@0.5 58.9, P@0.3 78.6, P@0.7 31.8, mIoU 53.3;
  - far targets: P@0.5 25.4;
  - hand P@0.5 92.6;
  - fingertip error 3.9 px;
  - median direction error 18.6°.
- Table 4: published numbers checked against the PDFs in `Literature review/` (VistaRef, PointVG-R, EgoPoint-Ground).
- The own VLM benchmark (former T1) is dropped; the published zero-shot results are used instead.
- Figures done: architecture (Canva), teaser, qualitative results, ownership examples.
- **224 px run** (`runs/img224/`), in the paper (§7.3, Tables 4 and 6):
  - P@0.5 62.4 (+3.5);
  - far P@0.5 35.9 (+10.5);
  - same 0.72M parameters.
  - Analysis: `docs/runs/img224.md`.
- **Ablation A1** (`runs/A1_no_copypaste/`), in the paper (§7.4, Tables 7 and 8):
  - with other hands in view: P@0.5 −4.1, no-hand frames kept −5.4;
  - clean images: unchanged.
  - Analysis: `docs/runs/A1_no_copypaste.md`.
- **Tools:**
  - `paper/scripts/compare_runs.py`: paired bootstrap of any run against the main run;
  - `paper/scripts/error_breakdown.py`: failure types and target-size bins;
  - `experiments/make_variant.py`: builds a Kaggle notebook for any variant.

### Running on Kaggle

- **Ablation A3, no ray conditioning** (`experiments/A3_no_roi_geometry/`, kernel `spatialnet-ablation-a3`).
- **320 px input** (`experiments/img320/`, kernel `spatialnet-train-320`):
  - uses the new private dataset `egopoint-spatialnet-320`, re-converted from the originals (labels identical to the 224 dataset);
  - tests whether a still larger input keeps helping.
- **When each finishes:**
  1. Download it to `runs/<name>/`.
  2. Run `python paper/scripts/compare_runs.py runs/main_run runs/<name>`.
  3. Write `docs/runs/<name>.md`.
  4. Fill in Table 6 (resolution) or Table 7 (ablations).

### Still to run

- Ablations A2 and A4 (section 3).
- Seeds 1 and 2 (section 4).

---

## 1. Decide the training recipe (do this first)

**Problem.** The main run stopped at epoch 125 through early stopping, before its 180-epoch learning-rate schedule finished.

**Rule.** The main run, the ablations and the seeds must all use the same recipe.

**Option A: keep the current recipe.**

- No re-run needed; the existing main run is reused.
- Ablations and seeds use the default settings.

**Option B: complete the schedule.**

- Use `--epochs 120 --patience 200`.
- Re-run the main model first, then update every main-run number in the paper.
- Costs one extra run (about 2 h), probably for a small accuracy gain.

**Decision (2026-10-07): Option A.** Keep the current recipe and do not re-run the main model. The ablations and seeds use the default settings, adding only their own flags.

---

## 2. T2: on-glasses measurements (most important)

**Goal.** Real latency, memory and energy on the glasses. The efficiency claim depends on this.

### Steps

1. **Export the model** (on your PC):
   - `python export_onnx.py --model_path runs/main_run/outputs_spatialnet/best_model.pth --out spatialnet.onnx --verify_image <any test image>`
   - Check that it prints parity OK.
   - Note the file size.
2. **Check operator support first:**
   - The model uses `grid_sample` (ONNX opset 17).
   - Convert with the glasses' runtime (e.g. QNN / SNPE / TFLite / ONNX Runtime Mobile).
   - Record whether it runs on the NPU, the GPU, or falls back to the CPU.
   - If `grid_sample` is not supported: run on the CPU and say so in the paper. Do not change the model now.
3. **Implement the same preprocessing as training:**
   - RGB, squash-resize to 160×160, divide by 255.
   - Normalise with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
   - Input tensor `[1, 3, 160, 160]`, float32 (or FP16 if the runtime uses it).
4. **Decode the outputs:**
   - Boxes and fingertip are normalised 0–1.
   - The gates are logits; apply a sigmoid and a 0.5 threshold.
   - The scene cascade is hand → pointing → target.
5. **Measure model latency:**
   - 20 warm-up runs, then 200 timed runs.
   - Record the median and the 90th percentile in ms.
6. **Measure end-to-end latency:**
   - From camera frame captured to result available or shown.
   - At least 100 frames; record the median and the 90th percentile.
7. **Measure sustained throughput:** run continuously for 60 s and record frames per second.
8. **Measure memory:** peak memory of the app (Android Studio profiler or `dumpsys meminfo`).
9. **Measure battery:**
   - 10 minutes of continuous inference.
   - Record the battery % drop, or the average current from BatteryManager.
10. **Check accuracy parity:**
    - Run 100 test images on the glasses and save the outputs.
    - Compare with PyTorch: share of images with the same scene, and the mean IoU of the target boxes between device and PyTorch.
11. **Take one photo** of the glasses running it, or a screenshot of the output (optional figure).

### Record

| Item | Value |
|---|---|
| Device, SoC, OS version | |
| Runtime, compute unit, precision | |
| Model latency, median / p90 | |
| End-to-end latency, median / p90 | |
| Throughput | |
| Peak memory | |
| Battery per 10 min | |
| Parity (same scene %, box IoU) | |

**Goes to:** §6.2 (pipeline paragraph), Table 8, abstract, introduction, conclusion.

**Done when:** every row of Table 8 has a number and §6.2 describes the pipeline.

---

## 3. T3: ablations (4 runs, one seed each)

**Goal.** Show that each design part matters.

- A4 is the important one: it is the same-size baseline without the anchored design.
- A3 decides whether the ray contribution can be claimed.

### Runs

| Run | `OUTPUT_DIR` | Extra train flags | Tests |
|---|---|---|---|
| A1 | `ablation_A1_no_copypaste` | `--copy_paste 0 --ownership_weight 0` | Ownership training |
| A2 | `ablation_A2_no_geometry_loss` | `--w_ray 0 --w_hand_axis 0` | Ray and hand-axis losses |
| A3 | `ablation_A3_no_roi_geometry` | `--roi_geometry none` | Ray conditioning of the target (main idea) |
| A4 | `ablation_A4_no_anchor_chain` | `--no_anchor_chain` | Anchored heads against a plain small model |

### Steps for each run

1. Copy `experiments/img224/` to a new folder, e.g. `experiments/A1/`.
2. In `make_notebook.py`, change:
   - the title and description in cell 0;
   - `OUTPUT_DIR` in cell 2;
   - the `cmd` line in cell 6: remove `--img_size`, add the flags from the table.
3. In `kernel-metadata.json`, set a new `id` (e.g. `mosharafapurbo/spatialnet-ablation-a1`), a new title, and the new notebook file name.
4. Run `python experiments/A1/make_notebook.py`.
5. Push with the `docs/kaggle.json` credentials only: `kaggle kernels push -p experiments/A1`.
6. Check `kaggle kernels status ...`.
   - If it fails with `ERRORED_MOUNTING_DATASET`, push again. That is a Kaggle-side error.
7. When it is complete: `kaggle kernels output ... -p runs/A1`.
8. Check the log for errors, and the training summary for the epoch count.

### After all four are done

- Give the `runs/` folders to Claude. Claude computes:
  - target mIoU, P@0.5, far P@0.5, hand mIoU, fingertip px and ownership accuracy;
  - **paired bootstrap** 95% intervals of the difference against the full model, over the same test images.
- Fill in Table 6 and write one sentence per row, only as strong as its interval.
- **If A3 is not clearly worse than the full model:** weaken the ray claim in the abstract, introduction and discussion.
- **If A4 is close to the full model:** the design adds little. Say so honestly and lean on the efficiency claim.

**Goes to:** Table 6, §7.3, §9 ("What the design contributes").

**Budget:** about 2 h per run, so about 8 GPU-hours in total.

---

## 4. T4: seeds (2 runs) — DONE 2026-10-08

**Done:** filled Table 4 (mean ± std over seeds 42, 1, 2: P@0.5 59.0±0.1) and the retraining row in Table 7; see `docs/runs/seeds.md`. The headline stays seed 42 (the deployed model).

**Goal.** Show that the main result is stable.

### Steps

1. Same procedure as T3, with `OUTPUT_DIR` set to `seed_1` and `seed_2`, and flags `--seed 1` and `--seed 2`.
2. Use the same recipe as the main run (see section 1).
3. Download to `runs/seed_1/` and `runs/seed_2/`.
4. Claude computes the mean ± std over seeds 42, 1 and 2 for every column of Table 4.

**Goes to:** Table 4 ("± std over 3 seeds" row); headline numbers can stay seed 42 or become the mean (decide then).

**Budget:** about 4 GPU-hours.

---

## 5. T5: real frames with other people's hands

**Goal.** Replace "synthetic only" ownership evidence with a real check. The current 99.7% is weak evidence.

### Steps

1. **Record 3–5 short clips with the glasses** in which:
   - another person's hand is visible (some clips with that person also pointing);
   - you, the wearer, point at objects.
2. **Consent:** get consent from everyone filmed, avoid faces, and keep the data private.
3. **Sample 100–150 frames**, spread over the clips, then run SpatialNet on them.
4. **Save each frame with the predicted hand box, fingertip and target drawn on it.** Claude can write this script on request.
5. **Two people rate every frame independently** (no box drawing needed):
   - **correct:** the predicted hand is the wearer's;
   - **ownership error:** the predicted hand is the other person's;
   - **missed:** no hand predicted although the wearer's hand is visible.
6. Claude computes:
   - ownership accuracy with a 95% Wilson interval;
   - agreement between the two raters (Cohen's κ).
7. **Choose 6 frames for the figure:** 4 correct, at least 1 ownership error, and 1 other failure.

**Goes to:** Table 7 (last row), Fig. 5, §7.4, ethics (consent sentence).

**Effort:** about one day.

---

## 7. Before submission (checklist)

User study (T6): removed from the paper and the plan on 2026-10-08. It is listed as future work in §9 (Limitations, "Interaction").

- [x] All red items except T5 resolved (2026-10-08). Only the T5 real-frame check is still red (§7.5 paragraph, Table 8 last row, Fig. 6, ethics sentence in §10).
- [x] Device: chip and vendor names removed from §6.2. Only generic hardware (SoC + MCU, 12 MP 105° camera, NPU, 160 mAh, <45 g) is given.
- [x] Dataset: §10 states use under the authors' access terms, with no redistribution of images or labels. Keep the Kaggle dataset private.
- [x] Split: §5 uses the released split as is, the same as VistaRef and PointVG-R.
- [x] CCS concepts: `\ccsdesc` entries are in `main.tex` (the XML is optional at review).
- [ ] After T5: delete the draft macros in `main.tex` (`\tbd`, `\xx`, `\phfig`), so that any forgotten placeholder breaks the build.
- [ ] Camera-ready only: add the code and weights link (§10 promises release upon publication).

---

## 8. Order and budget

| Order | Task | GPU-hours | Your time |
|---|---|---|---|
| 1 | Decide the recipe (section 1) | 0–2 | 5 min |
| 2 | T2 on the glasses | — | 1–2 days |
| 3 | T3 ablations A1–A4 | about 8 | about 1 h of setup |
| 4 | T4 seeds | about 4 | about 30 min |
| 5 | T5 real frames | — | about 1 day |
| 8 | 224 px run (running) | about 3 | — |

The Kaggle quota is about 30 GPU-hours a week, so T3 and T4 fit in one week.
