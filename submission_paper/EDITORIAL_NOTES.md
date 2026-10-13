# Editorial and evidence notes

## Deliverable

The manuscript is written as a compact, venue-neutral anonymous research submission. It does not use IMWUT branding or require a user study. The only incomplete manuscript passage is the red T5 block on real multi-hand footage. The original manuscript, code, datasets and run outputs were preserved.

The central story is a measured compact operating point, not superiority over large models:
- a 722K-parameter image-only localiser;
- improved strict localisation from instance conditioning under the existing recipe;
- synthetic distractor robustness from the training package;
- a clear input-resolution/compute trade-off;
- a negative result separating direction supervision from target-box improvement.

## Numerical provenance

The primary metric source is each run's evaluation/predictions/test.csv, supplemented by evaluation/metrics.json and saved comparison_vs_main_run.json.
Synthetic stress metrics use predictions/test_multihand.csv. Selected rows are recorded in evidence.json.
Paired intervals are the saved 1,000-resample, seed-0 percentile intervals produced by paper/scripts/compare_runs.py.
Full-model seed summaries use seeds 42, 1 and 2. The sample standard deviation of mIoU from unrounded saved per-image predictions is 0.4483 percentage points, printed as 0.4; the prior manuscript's 0.5 reflects a coarser summary. No predictions or experiment results were changed.
Paired-image uncertainty is not training-seed uncertainty. Single-seed ablations, differing checkpoint epochs and bundled changes are explicitly qualified.

No new model training or aggregate evaluation was performed. The audit reads existing CSV/JSON files. A single optional CPU inference check was attempted during example preparation; final figure overlays consistently use saved predictions and are verified directly against stored annotations.

## Requested comparison table

Retained: PointVG-R and Qwen supervised/zero-shot comparisons, hand-box-assisted SV-CoT and Qwen3-VL rows, the main SpatialNet model, three-seed summary, and 224/320-pixel variants.
Removed at the author's request: VistaRef and all simple-reference rows.
The short diagnostic fingertip-baseline paragraph remains to motivate target extent.

Published PointVG-R rows were checked against Literature review/Point-GR.pdf, Table 1.
Hand-box-assisted rows were checked against Literature review/Visual Grounding.pdf, Table 4, POG columns.
SV-CoT's exact parameter total is not established by that table, so the column shows not reported rather than an unsupported >1B estimate.
Published protocols and no-target denominators are not assumed identical to the converted EgoPoint evaluation.
VistaRef remains cited in related work but is absent from the comparison table.

## Terminology and interpretation

- "Overlap" means annotated fingertip lies inside a 2D target box, not physical touch.
- "Large gap" means normalized image-plane distance >0.08, not physical distance.
- Direction supervision is fingertip-to-target-centre, not independent 3D finger pose.
- A4 retains local size heads and target ray conditioning; it is not a wholly independent-head baseline.
- A1 removes both pasting and negative weighting; A2 removes both geometry losses.
- Synthetic stress-test scores include 227 positive images without a successful paste.
- Learned-ray probes are existing exploratory observations consistent with proximity cues, not proof of collapse or a causal explanation.
- Prototype latency, incremental memory and coarse battery observations are reported with their measurement scope. They do not establish sustained throughput, total runtime memory, battery life or general energy superiority.

## Figure selection

Figure 1 contains deliberately selected correct examples: banana, truck, traffic light and key.
Figure 5 contains deliberately selected complex failures: tableware, bookshelf, garden and harbour.
The captions do not imply that successful examples represent average accuracy or that failures occur only in complex scenes.
example_provenance.json records all eight test IDs, predicted and annotated target boxes, and displayed IoUs.
Original dataset photos are shown in the same normalized square coordinate frame used by the model. Annotations are not relabelled to fit predictions.
Direction is omitted from the photo overlays for clarity; the method and architecture explain the direction output.

## Submission use

The paper is technically complete apart from the explicitly requested T5 insertion. Dashes in the comparison table mean unreported/not applicable, not missing author work. No additional experiment, user study, or author query is embedded as an unfinished task elsewhere in the manuscript.

## Abstract and introduction rewrite (October 2026)

The abstract and introduction now open with visual grounding and embodied reference, the way VistaRef and PointVG-R open, before narrowing to smart glasses. They then compare SpatialNet with large models in terms of scale and their reported accuracy. Every new external claim is cited and was checked against its source in October 2026:
- Market growth: IDC press release, 23 March 2026 (XR shipments +44.4% in 2025, driven primarily by smart glasses; glasses without displays are the majority of XR shipments). Counterpoint Research, 21 September 2026 (AI-glasses shipments +263% YoY in H1 2026; display-less models are 96% of the total).
- Glasses constraints: Yuan et al., arXiv:2609.19793. Commercial glasses have 200-300 mAh batteries, less than 1/20 of a smartphone's. Multimodal foundation models exceed current wearable processors, so they are delegated to the cloud. Lee et al., arXiv:2512.16045, attribute stringent power constraints to weight and battery limits.
- On-device memory: Liu et al., MobileLLM (ICML 2024). Phones have 6-12 GB DRAM, one app should use at most about 10% of it, and a 7B model with 8-bit weights is prohibitively expensive on a phone.
- Model sizes: VistaRef's base variant has 226.01M parameters (VistaRef Table 5). PointVG-R is built on Qwen2.5-VL-7B. SV-CoT is instantiated on Qwen3-VL. All three were checked against the PDFs in Literature review/.
- Derived arithmetic:
  - 7e9 x 2 bytes = 14 GB of 16-bit weights.
  - 7e9 / 722,104 = about 9,700, reported as "roughly 10^4".
  - 0.92 / 0.231 GFLOPs = about 4x for the 320-pixel input.
- Sub-million precedent on glasses: TinyissimoYOLO (ECCV 2024 Workshops) does generic detection, not reference resolution.

The comparison with large models still frames published numbers as context for the accuracy regime, not as a controlled ranking. PointVG-R's higher accuracy (83.6 vs 59.0 P@0.5; 74.5 vs 31.5 P@0.7) is stated explicitly. With the longer introduction and eight new references, the manuscript now runs 11 pages. The \balance trigger moved to huang2016pointing.

## Benchmark additions (October 2026)

Source: `docs/BENCHMARK_RESULTS (2).md` (status 2026-10-11). Its raw result folders are not in this repository, so these numbers are not reconstructed by the audit script or recorded in evidence.json.
- Added: compact detectors at matched compute (YOLO26n with COCO, ImageNet and no pretraining; NanoDet-Plus-m), the end-to-end score, the duplicate check, the backbone swap, and the full-test ray-branch probe.
- The benchmark's "SpatialNet v2" (wider size window plus box refiner) is deliberately not reported, at the author's request. The 160-pixel v1 checkpoint stays the reference.
- The benchmark's paired intervals against detectors are all for v2. SpatialNet-vs-detector differences are therefore point estimates, and no significance is claimed for them.
- Pasted-hand no-hand rejection is not compared across models, because the benchmark reports it only for v2, with val-tuned thresholds.
- Duplicate check: the A3 and A4 P@0.7 effects lose significance and are now qualified in the text.
- New red placeholder: the detector training recipe and per-image box-selection rule (protocol section).
- New references: YOLO26 and NanoDet-Plus are cited as software repositories. Verify the YOLO26 author list and year against the Ultralytics citation file before submission.
