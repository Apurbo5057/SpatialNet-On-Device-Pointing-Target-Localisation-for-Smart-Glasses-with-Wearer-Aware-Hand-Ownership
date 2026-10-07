# Evaluation

```bash
python test_evaluate.py --model_path <out>/best_model.pth --data_dir <dataset> \
    --output_dir <out>/evaluation --multihand --calibrate_thresholds
```

Evaluates one checkpoint on the `test/` split and saves every number, table and figure, plus one row per test image. Any metric can be recomputed later without the model. The Kaggle notebook runs exactly this command.

## Test conditions

| Condition | What it is | Flag |
|---|---|---|
| `test` | Test images as stored, gate thresholds 0.5 | always |
| `test_calibrated` | Same predictions, with thresholds chosen on **validation**: for each gate, the highest threshold ≤ 0.5 that misses no positive on validation | `--calibrate_thresholds` |
| `test_multihand` | Test images with 1–2 pasted hands at fixed, seeded positions. The hands come from other test images; 30% are the wearer's own hand re-pasted. Labels unchanged: only the wearer's hand counts. | `--multihand` |

Test images are only resized to 160 × 160 and normalised, with no augmentation.

> **Limitation of `test_multihand`.** The pasted hands are made by the same procedure as the training augmentation: same placement rules (centre in the top 65% of the image), same rotation and flipping, and same self-paste of the wearer's hand. The model has seen this kind of distractor during training, so the ownership numbers measure robustness to that synthetic distribution and may overstate robustness to real scenes. EgoPoint contains no real frames with other people's hands. A small set of real egocentric frames with other people's hands, labelled with the wearer's hand only, would make the wearer-awareness claim much stronger.

## How a scene is decided

The model outputs three gate probabilities. With thresholds t (0.5 by default), they are read in order:

1. hand < t → **No hand** (H P T = 000)
2. pointing < t → **Non-pointing** (100)
3. target < t → **Pointing, no target** (110)
4. otherwise → **Pointing at target** (111)

The ground-truth scene comes from the label's H P T flags.

## Metrics

**Scene classification** (all images)

| Metric | Meaning |
|---|---|
| Accuracy | Share of images with the correct scene, with a 95% Wilson confidence interval |
| Macro-F1 | Mean F1 over the four scenes. Each scene counts equally, however rare. |
| Balanced accuracy | Mean recall over the four scenes |
| Per scene | Support (true count), precision, recall with 95% CI, F1 |
| Confusion matrix | Rows = true scene, columns = predicted scene |

**Gates**: each gate is scored only on the images where it applies.

| Gate | Images scored | Positive means |
|---|---|---|
| Hand present | all | H = 1 |
| Pointing | true hand images | P = 1 |
| Target present | true pointing images | T = 1 |

For each gate the evaluation reports images, positives, threshold, accuracy, recall, **missed** (false negatives), **false alarms** (false positives), plus two threshold-free measures. **AUROC** is the chance a random positive scores higher than a random negative (0.5 = chance). **AP** is the area under the precision–recall curve, computed as scikit-learn does.

**Localisation** is measured on every image that has the box or point in its label, whatever scene the model predicted.

| Measure | Subsets | Reported |
|---|---|---|
| Hand IoU | Non-pointing, Pointing no target, Pointing at target, All hand-present | Mean, 95% CI of the mean, median, % of images with IoU ≥ 0.3 / 0.5 / 0.7 / 0.75 |
| Target IoU | Pointing at target: all, and by fingertip–target gap | Same as hand IoU |
| Fingertip error | Pointing no target, Pointing at target, All pointing | Mean px, 95% CI, median, 90th percentile, % < 5 px, % < 10 px |
| Ray error | Same as fingertip | Mean degrees, 95% CI, median, 90th percentile, % < 15°, % < 30° |

- **IoU**: overlap ÷ union of the predicted and true box (0–1). "IoU ≥ 0.5" is the share of images localised at least that well.
- **Pixels** are at the 160 × 160 model input; 1 px ≈ 0.6% of the image width.
- **Ray error** is the angle between the predicted pointing direction and the fingertip → target-centre direction. For "Pointing, no target", the reference is hand centre → fingertip.
- **Gap** is the distance from the fingertip to the target box, as a fraction of image size. On-object = 0 (touching), near ≤ 0.08, far > 0.08. The gap detail table adds fingertip error, ray error, and pointing images misread as no hand or non-pointing, per gap.
- **95% CI of a mean**: percentile bootstrap, 1,000 resamples, fixed seed.

**Ownership** (`test_multihand` only)

| Metric | Meaning |
|---|---|
| Ownership accuracy | Share of hand images where the predicted hand centre is nearer the wearer's true hand than any pasted hand, with 95% CI |
| Hand centre inside a distractor | Share where the predicted hand centre falls inside a pasted hand. Lower is better. |
| Ownership on pointing images | The same accuracy, on pointing-at-target images only |
| Still "No hand" | Share of no-hand images with pasted hands still classified as No hand |

**Headline comparison**: accuracy, macro-F1, missed and false alarms per gate, hand and target IoU (mean and ≥ 0.5), and fingertip error, side by side for every condition run.

**Efficiency**

| Metric | Meaning |
|---|---|
| Parameters | Total, backbone and heads |
| Compute | GFLOPs for one 160 × 160 image. Counts convolutions, linear layers and matrix products; sampling operations are not counted. |
| Size | Saved state dict in FP32, FP16 and INT8 (INT8 quantises linear layers only) |
| Latency | Median and 90th percentile of 50 forward passes after 10 warm-up runs, batch 1, PyTorch eager mode, on the machine running the evaluation (CPU with 1 thread, CPU with all threads, GPU if present). This is not on-device latency. |

## Saved files (`<out>/evaluation/`)

| Path | Contents |
|---|---|
| `metrics.json` | Everything: checkpoint (path, epoch, val loss), environment, and for each condition the thresholds, classification, gates, localisation and ownership, plus validation metrics, efficiency and runtime |
| `summary.md` | Every table, in the order printed, as Markdown |
| `tables/*.csv` | Each table as CSV (list below) |
| `predictions/*.csv` | One row per image (columns below) |
| `figures/` | PDF for the paper, PNG for preview |

**Tables**: `test_scene_classification`, `test_confusion_matrix`, `test_gates`, `test_hand_iou_by_scene`, `test_target_iou`, `test_fingertip_error`, `test_ray_error`, `test_gap_detail`; for calibrated thresholds, `test_calibrated_scene_classification`, `_confusion_matrix` and `_gates`; for pasted hands, `test_multihand_scene_classification`, `_confusion_matrix`, `_gates`, `_hand_iou_by_scene` and `_ownership`; plus `comparison` and `efficiency`.

**Predictions**: `test.csv`, `val.csv` (with `--calibrate_thresholds`) and `test_multihand.csv` (with `--multihand`).

| Column | Meaning |
|---|---|
| `name` | Image file name without extension |
| `gt_H`, `gt_P`, `gt_T`, `gt_scene` | Label flags and scene |
| `pred_scene`, `correct` | Predicted scene at 0.5 thresholds; 1 if it matches |
| `hand_prob`, `pointing_prob`, `target_prob` | Raw gate probabilities. Re-threshold these to get any other operating point. |
| `hand_iou`, `target_iou`, `tip_err_px`, `ray_err_deg` | Per-image errors, empty when the label has no such box or point |
| `tip_target_gap`, `gap_bin` | Fingertip–target distance and its bin (pointing-at-target images) |
| `n_distractors`, `owner_correct`, `pred_on_distractor` | Pasted hands and ownership result (multi-hand only) |
| `pred_hand_*`, `pred_tip_*`, `pred_target_*` | Predicted boxes (cx, cy, w, h) and fingertip (x, y), normalised to 0–1 |

**Figures**

| File | Shows |
|---|---|
| `confusion_matrix` (`_calibrated`, `_multihand`) | Row-normalised confusion matrix with counts |
| `gate_roc`, `gate_pr` | ROC and precision–recall curves of the three gates, with AUROC / AP |
| `iou_success_rate` | Share of images above each IoU threshold, for hand and target |
| `fingertip_error_cdf`, `ray_error_cdf` | Cumulative error distributions per pointing scene |
| `target_iou_by_gap` | Mean target IoU and IoU ≥ 0.5 per gap bin |
| `qualitative_examples` (PNG) | Sample predictions per scene, ground truth dashed, with H P T flags in titles |

## Reporting guidance

- **The paper uses `best_model.pth`** (lowest validation loss), fixed before looking at test results. Evaluate the test split once.
- **With EgoPoint, only two scenes exist:** No hand and Pointing at target (`dataset_egopoint.md`). The Non-pointing and Pointing-no-target rows show 0 or "-". They are left out of macro-F1 and balanced accuracy, so no code change is needed. Their predicted columns still count errors when the model wrongly predicts those scenes.
- **The pointing and target gates have no negative examples in EgoPoint.** Their accuracy is trivially 100% and AUROC / AP cannot be computed. Report the hand gate, scene accuracy (No hand vs Pointing at target), localisation and ownership; do not claim pointing-gate or target-gate performance.
- State the `test_multihand` limitation above wherever ownership results are reported.
- Report thresholded rates (IoU ≥ 0.5 and the others) together with the mean. Mean IoU is pulled down by a few failures; the success rates show how often localisation works.
- Report rare scenes (Pointing, no target) with their confidence intervals; a small support gives a wide interval.
