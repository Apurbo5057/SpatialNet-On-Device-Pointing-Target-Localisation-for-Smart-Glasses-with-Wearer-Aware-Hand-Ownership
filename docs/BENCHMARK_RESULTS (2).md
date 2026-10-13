# SpatialNet benchmark results

Status: 2026-10-11. Plan: `docs/benchmarking_plan.md`. Raw outputs are in the result folders in this directory.

**Test protocol:**
- **Images:** EgoPoint test split, 2,604 images with a pointed-at target and 425 with no hand. Every model is scored by the same code (`bench_common.py`) on the same images.
- **Statistics:** differences are paired, with a 95% bootstrap interval.
- **Margin:** "not worse" means the lower end of the interval is above −3 points of target P@0.5 (fixed before running).
- **Duplicates:** the duplicate-free test set (141 images removed) gives the same conclusions everywhere unless stated otherwise.
- **Two kinds of score:**
  - target metrics (P@0.5, mIoU, ...) use the 2,604 target images and score every target box, whether or not the hand gate fired;
  - the **end-to-end score** uses all 3,029 test images and also counts missed hands and false alarms (see "End-to-end score" below).

## Bottom line

1. **SpatialNet picks the right target best of all models**: the predicted centre lands inside the true target box in 84% of images, against 82–83% for every detector.
2. **Its weakness is box size**, not target choice. Detectors draw tighter boxes, especially for large or far targets (P@0.7 and large-gap P@0.5).
3. **v2** (wider size window plus box refiner) improves on v1: **+1.9 P@0.5** and **+4.6 P@0.7**, both significant.
4. **Against detectors at the same compute (~0.23 GFLOPs), v2 still trails:**
   - COCO-pretrained YOLO26n by −4.2 points;
   - NanoDet-Plus-m by −6.8 points;
   - YOLO26n with the **same ImageNet pretraining** by only −2.4 points, just outside the 3-point margin;
   - **YOLO26n from scratch: v2 is not worse** (+0.9).
5. **SpatialNet is the only model whose accuracy does not drop with 1–2 pasted distractor hands:** +1.4 points against −3.6 to −4.6 for the detectors. It also rejects no-hand frames best (98.8%).
   - Caveat: the test is synthetic, and only SpatialNet was trained with pasted hands.
6. **The end-to-end score over all 3,029 test images tells the same story:** v2 65.3% against 68.9% (YOLO COCO), 67.4% (YOLO ImageNet), 63.8% (YOLO scratch) and 71.1% (NanoDet).
7. **Size:** v2 is 0.83M parameters (1.8 MB in FP16). That is 1.4× smaller than NanoDet-Plus-m and 3× smaller than YOLO26n.
8. **Backbone choice does not matter.** MobileNetV3/V4 give no significant P@0.5 change while making the model 2–3× larger, so ShuffleNetV2 0.5× is the right choice.

## Master table (test set, 2,604 target images)

| Model | Input | Params | Size FP32 / FP16 | GFLOPs | **P@0.5** | P@0.7 | mIoU | Large-gap P@0.5 | Hand mIoU | Centre in target | P@0.5 change, pasted hands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SpatialNet v1** (paper) | 160 | 0.72M | 3.0 / 1.6 MB | 0.231 | 58.9 | 31.8 | 53.3 | 25.4 | 73.7 | **84.8%** | **+1.2** |
| **SpatialNet v2** | 160 | 0.83M | 3.5 / 1.8 MB | 0.231 | **60.8** | 36.4 | 54.7 | 25.4 | 77.6 | 84.1% | **+1.4** |
| SpatialNet v1, MobileNetV3-S backbone | 160 | 1.57M | 6.4 / 3.2 MB | 0.255 | 58.1 | 31.3 | 52.3 | 31.1 | 73.2 | — | +2.4 |
| SpatialNet v1, MobileNetV4-Conv-S backbone | 160 | 1.99M | 8.1 / 4.1 MB | 0.398 | 59.8 | 30.7 | 52.8 | 30.1 | 73.6 | — | +1.2 |
| SpatialNet v1 | 224 | 0.72M | 3.0 / 1.6 MB | 0.453 | 62.4 | 37.1 | 54.8 | 35.9 | 74.9 | 83.7% | +1.5 |
| SpatialNet v1 | 320 | 0.72M | 3.0 / 1.6 MB | 0.923 | 65.1 | 40.3 | — | 34.9 | — | — | — |
| YOLO26n, COCO start | 128 | 2.50M | 9.6 / 4.8 MB | 0.225 | 65.0 | 47.0 | 58.6 | 39.7 | 81.6 | 82.2% | −4.0 |
| YOLO26n, ImageNet start | 128 | 2.50M | 9.6 / 4.8 MB | 0.225 | 63.2 | 42.4 | 56.8 | 36.8 | 81.1 | 82.3% | −3.6 |
| YOLO26n, from scratch | 128 | 2.50M | 9.6 / 4.8 MB | 0.225 | 59.9 | 39.1 | 54.7 | 31.6 | 80.6 | 81.6% | −4.6 |
| YOLO26n, COCO start | 160 | 2.50M | 9.6 / 4.8 MB | 0.352 | 66.9 | 48.8 | 60.3 | 42.1 | 82.1 | 83.5% | −3.0 |
| YOLO26n, COCO start | 192 | 2.50M | 9.6 / 4.8 MB | 0.507 | 69.9 | 53.3 | 62.8 | 43.5 | 82.9 | 84.7% | −2.6 |
| **NanoDet-Plus-m**, COCO start | 160 | 1.17M | 4.7 / 2.4 MB | 0.218 | **67.6** | **49.5** | **59.8** | **40.7** | 81.5 | 82.9% | −4.5 |

**Notes on the table:**
- **Sizes:**
  - SpatialNet: serialized state dicts;
  - YOLO26n and NanoDet: the exported ONNX files, with FP16 taken as half of FP32;
  - NanoDet: inference parameters only, since its 3M-parameter training branch is dropped.
- **GFLOPs** were counted with the same `torch` FLOP counter for every model.
- **"Centre in target"** is the share of images whose predicted target centre falls inside the true target box. It measures target *choice*, independent of box size.
- **Speed:** latency was not compared across models (B4P is still open). In PyTorch eager mode on Kaggle, v1 / v2 take 25.7 / 27.6 ms on 1 CPU thread and 15.9 / 17.8 ms on a T4 GPU.

## End-to-end score (all 3,029 test images)

**What counts as correct:**
- on a target image: the hand gate fires, the model predicts "pointing at target", **and** the target box has IoU ≥ 0.5;
- on a no-hand image: the model correctly says "no hand".

Every model's hand-gate threshold is tuned on val by the same rule. Results are in `runs/benchmarks/end_to_end_accuracy.csv`.

| Model | GFLOPs | Params | P@0.5 (2,604) | P@0.5 counting the gate | No-hand correct (425) | **End-to-end (3,029)** | Without duplicates |
|---|---|---|---|---|---|---|---|
| SpatialNet v1 160 | 0.231 | 0.72M | 58.9 | 58.2 | 97.2 | 63.6 | 62.4 |
| **SpatialNet v2 160** | 0.231 | 0.83M | 60.8 | 60.0 | 98.1 | **65.3** | 64.0 |
| SpatialNet v1 160, MobileNetV3-S | 0.255 | 1.57M | 58.1 | 57.5 | 96.9 | 63.1 | 61.8 |
| SpatialNet v1 160, MobileNetV4-Conv-S | 0.398 | 1.99M | 59.8 | 59.0 | 98.6 | 64.5 | 63.2 |
| SpatialNet v1 224 | 0.453 | 0.72M | 62.4 | 61.9 | 96.5 | 66.8 | 65.7 |
| SpatialNet v1 320 | 0.923 | 0.72M | 65.1 | 64.5 | 98.4 | 69.3 | 68.4 |
| YOLO26n 128, COCO | 0.225 | 2.50M | 65.0 | 64.2 | 97.9 | 68.9 | 67.9 |
| YOLO26n 128, ImageNet | 0.225 | 2.50M | 63.2 | 62.6 | 96.9 | 67.4 | 66.6 |
| YOLO26n 128, scratch | 0.225 | 2.50M | 59.9 | 58.6 | 95.3 | 63.8 | 62.9 |
| YOLO26n 160, COCO | 0.352 | 2.50M | 66.9 | 66.4 | 97.7 | 70.8 | 69.9 |
| YOLO26n 192, COCO | 0.507 | 2.50M | 69.9 | 69.6 | 98.1 | 73.6 | 72.6 |
| NanoDet-Plus-m 160, COCO | 0.218 | 1.17M | 67.6 | 66.9 | 96.5 | 71.1 | 70.1 |

**v2 minus each model, end-to-end** (paired 95% interval; full test set, then without duplicates):

| Compared with | Full test set | Without duplicates | Verdict (3-point margin) |
|---|---|---|---|
| SpatialNet v1 160 | +1.7 [+0.3, +3.1] | — | v2 better |
| YOLO26n 128, scratch | +1.6 [−0.1, +3.2] | +1.1 [−0.5, +2.9] | v2 not worse |
| YOLO26n 128, ImageNet (equal pretraining) | −2.1 [−3.7, −0.5] | −2.6 [−4.2, −0.9] | YOLO better; just misses the margin |
| YOLO26n 128, COCO | −3.6 [−5.4, −2.0] | −3.9 [−5.6, −2.2] | YOLO better |
| NanoDet-Plus-m 160, COCO | −5.7 [−7.3, −4.2] | −6.1 [−7.8, −4.5] | NanoDet better |

**What this adds:**
- Counting the gate costs every model about 0.5–1.3 points against plain P@0.5. All models miss few hands, so the rankings do not change.
- The gap to YOLO26n with ImageNet pretraining shrinks slightly: −2.1 end-to-end against −2.4 on P@0.5.
- The paper can report this single number for "found the target and raised no false alarm", next to P@0.5.

## SpatialNet v2 against every other model (target P@0.5, v2 − other)

| Compared with | GFLOPs | Params | v2 − other [95% CI] | Verdict |
|---|---|---|---|---|
| SpatialNet v1 160 | 0.231 | 0.72M | **+1.9** [+0.3, +3.7] | **v2 better** |
| YOLO26n 128, from scratch | 0.225 | 2.50M | **+0.9** [−1.0, +2.8] | **v2 not worse** |
| YOLO26n 128, ImageNet start (equal pretraining) | 0.225 | 2.50M | −2.4 [−4.2, −0.5] | YOLO better (narrowly misses the margin) |
| YOLO26n 128, COCO start | 0.225 | 2.50M | −4.2 [−6.1, −2.3] | YOLO better |
| YOLO26n 160, COCO start (same input size) | 0.352 | 2.50M | −6.1 [−8.1, −4.3] | YOLO better |
| NanoDet-Plus-m 160, COCO start | 0.218 | 1.17M | −6.8 [−8.7, −4.8] | NanoDet better |

**With pasted distractor hands** (synthetic stress test), v2 is not worse than any model at matched compute:
- v2 reaches 62.2% against 61.0% (YOLO COCO), 59.6% (YOLO ImageNet) and 63.2% (NanoDet);
- v2 keeps **98.8%** of no-hand frames as no-hand, against 92.5–96.7% for the detectors.

## Results per benchmark

### B1: test set without duplicates
- **Duplicates found:** 141 test images (4.7%) are byte-identical to train or val images; 362 have an exact or near copy.
- **Effect on headlines:** main-run P@0.5 goes from 58.9 to 57.2, which stays inside its interval. All other headlines also stay inside their intervals.
- **Conclusions that flip:** A3's P@0.7 gain (+1.9 becomes +1.6 [−0.2, 3.3]) and A4's P@0.7 drop (−2.8 becomes −1.7 [−3.5, 0.0]) lose significance, so **reword them in the paper**.

### B2: gate scope
EgoPoint contains only "pointing at target" and "no hand" images, so the pointing and target gates have no negatives. **Only the hand gate is evaluable:** AUROC 0.998, with 15 misses and 28 false alarms at threshold 0.5.

### B6: ray-probe outputs (for `results.tex:90`)
- **Removing the ray-geometry branch at inference:** P@0.5 falls from 58.9 to 50.6 overall, but on large-gap targets it goes from 25.8 to 27.8.
- **Learned prior:** it correlates more with distance to the fingertip (ρ −0.38) than with the ray line (−0.27).

### B8 and B8b: YOLO26n at matched compute, and pretraining
- **COCO pretraining explains about 40% of YOLO's lead:** 65.0 drops to 63.2 with an ImageNet start, and to 59.9 from scratch.
- **What remains at equal pretraining:** a −2.4 point gap, concentrated in tight boxes (P@0.7 −6.0) and large-gap targets (−11.5).
- **From scratch,** the best epoch was 149 of 150, so its number may be slightly low.

### B11: NanoDet-Plus-m (same compute, 1.4× the parameters)
- **Strongest baseline:** 67.6 P@0.5. It wins on box quality: P@0.7 +13.1, large-gap +15.3 over v2.
- **Robustness:** it loses 4.5 points with pasted hands and keeps only 92.5% of no-hand frames.
- **Pretraining:** it started from COCO. Its equal-pretraining version was not run.

### B10: backbone swap (v1 recipe, only the backbone changed)

| Backbone | Params | Size FP16 | GFLOPs | P@0.5 (Δ vs ShuffleNetV2) | Large-gap P@0.5 | Fingertip error |
|---|---|---|---|---|---|---|
| ShuffleNetV2 0.5× (paper) | 0.72M | 1.6 MB | 0.231 | 58.9 | 25.4 | 3.9 px |
| MobileNetV3-Small | 1.57M | 3.2 MB | 0.255 | 58.1 (−0.8 [−2.7, +1.0]) | 31.1 (+5.7 [−0.5, +12.0]) | 3.5 px |
| MobileNetV4-Conv-Small | 1.99M | 4.1 MB | 0.398 | 59.8 (+0.8 [−0.9, +2.8]) | 30.1 (+4.8 [−1.0, +10.5]) | 3.5 px |

- **No significant P@0.5 gain,** for 2.2–2.8× the parameters. ShuffleNetV2 0.5× is the efficient choice.
- **Possible effect on far targets:** both larger backbones lean better on large-gap targets (+5) and fingertips (−0.4 px), but the large-gap gain is not significant.
- **Caveat:** both runs early-stopped at epoch 89 (best epoch 64), earlier than the main run (125/100).

### SpatialNet v2 (box-sizing fix)

| | v1 | v2 |
|---|---|---|
| P@0.5 / P@0.7 | 58.9 / 31.8 | **60.8 / 36.4** |
| Near-target P@0.5 | 49.2 | **54.9** (+5.7) |
| Hand mIoU | 73.7 | **77.6** (+3.9) |
| Box-width error (median \|log ratio\|) | 0.188 | 0.177 (YOLO: 0.136) |
| Large-gap P@0.5 | 25.4 | 25.4 (no change) |
| Params / GFLOPs | 0.72M / 0.231 | 0.83M / 0.231 |

**The fix helps but closes only ~20% of the box-size gap. Large or far targets did not improve,** because the refiner reads only fine stride-8 features.

## What this means for the paper

**Claims that are supported:**
- SpatialNet v2 is within 2.4 points of YOLO26n at equal (ImageNet) pretraining and the same compute, with 3× fewer parameters, and is not worse than YOLO26n trained from scratch.
- It chooses the correct target object more often than any detector.
- It is the only model whose accuracy does not drop under synthetic distractor hands.

**Must be reported honestly:**
- COCO-pretrained detectors draw tighter boxes. NanoDet-Plus-m is 6.8 points better at the same compute with 1.4× the parameters.
- The A3 and A4 P@0.7 claims are not robust to duplicate removal.
- Robustness is so far shown only on synthetic pasted hands.

## Still open

| Item | Why it matters |
|---|---|
| **B5a, EgoHands (real other people's hands)** | Needed to back the robustness claim on real video. Needs a dataset copy with `polygons.mat`, or the Roboflow "specific" export (the loader would need adapting). |
| **YOLO26n trained with pasted hands** | Shows whether the robustness comes from SpatialNet's design or from its training augmentation |
| **B3/B4P, laptop latency** (ONNX files ready) | Speed comparison against NanoDet, which is nearly as small |
| B9, MediaPipe pipeline | Training-free baseline |
| Optional: SpatialNet v3 (multi-scale box refiner) | Targets the large-box gap |
| B4, B5b (on the glasses) | Deferred until the device is available |
