# SpatialNet paper, revised draft (IMWUT)

This folder holds the working draft for the November IMWUT submission. It is a revised copy of `paper/`, which is unchanged. It applies the writing changes from the three simulated reviews in `docs/` and adds the experiments run on Kaggle on 14 October 2026.

## Status

- `main.pdf` builds cleanly: 32 pages, with no LaTeX errors or overfull boxes.
- **Title:** *SpatialNet: Lightweight Pointing-Based Visual Grounding on Smart Glasses*. "Wearer-Aware Hand Ownership" is gone: the ownership evidence is synthetic, and the robustness comes from the training recipe, which works for a detector too.
- **All experiments are done.** No more Kaggle runs are needed.
- **Red placeholders remain in three places only,** and all three need data from you:
  - the device measurements (still written as single observations);
  - the real-frame check on the glasses (§7.8, Table 13, Fig. 5, §9);
  - the training recipe of your original benchmark (§5.2).

**Where the paper stands** (simulated IMWUT reviewer, same scale as the three reviews in `docs/`):

| State | Score | Likely decision |
|---|---|---|
| Now: recognition test and all minor fixes done | about 6.5 | Major revision |
| + device data and the real-frame check (step 1 below) | about 7–7.5 | Minor revision |
| + a small user study on the glasses and a demo video (step 4 below) | about 8 | Accept, possibly strong accept |

No list guarantees a strong accept, because reviewers differ. These items remove the remaining reasons a reviewer would score lower.

## What's left

### Before submitting (1 November)

**1. Get the device data.** The request is `docs/Samsung_Device_Measurement_Request.md`.

- **Must:** A (deployment check), B (latency), C (memory), D (power), G (real frames). G now needs only the wearer:
  - G1: 120–150 frames pointing at everyday objects in 4–6 places;
  - G2: 30–40 frames with no hand;
  - a log of which object the wearer meant in each G1 frame.
- **Optional:** E (detectors on the glasses; needs your ONNX exports of YOLO26n at 128 px and NanoDet-Plus-m at 160 px), F (extra power modes), G3 (a companion's hand in view), H (extras for a strong accept; see step 4).

When the data comes back, send me the files and I'll write the text, or fill these yourself:

| Data | Where in the paper |
|---|---|
| A: deployment check | One sentence in §6.2: the on-glasses model gives the same outputs as the evaluated one (state the largest difference). |
| B, C: latency, throughput, memory | §6.2 (what was timed, over how many frames), §7.7 and Table 12 |
| D: power per mode (mean ± SD), M2 − M1, streaming | §7.7 "Battery" paragraph and Table 12. Then replace "single one-minute observations" in §6.2, §7.7, §8 and Table 12. |
| G: real frames | §7.8 (all red text), Table 13, Fig. 5 (six frames, at least one failure), the consent text in §9, and one clause in the abstract (marked by a `%%` comment) |
| G3, if recorded | Ownership accuracy on those frames, at the `%%` comment at the end of §7.6 |

**To score G:** draw the hand box and the meant object's box on each frame (two annotators if possible), without looking at the model's outputs. Then compute, each with a 95% Wilson interval:

- hand P@0.5 and target P@0.5 (IoU ≥ 0.5);
- centre in: the predicted target centre lies inside the annotated box;
- no hand kept: G2 frames whose hand probability is below 0.5.

The model's boxes are normalised coordinates, which are the same in the squashed 160 × 160 input and in the full portrait frame. So draw your boxes on the full frame and normalise them by its width and height.

**If G cannot be done in time,** delete §7.8 with Table 13 and Fig. 5, and the clause "where we also check its accuracy on the unit's own camera frames" in contribution 1 (marked by a `%%` comment). Also delete the §9 consent placeholder. Then add one sentence to the limitations saying that accuracy on the glasses' own frames was not measured.

**2. Write your benchmark's training recipe.** Table 5 comes from your benchmark. Fill the red placeholder in §5.2: classes, epochs, augmentation, how one hand box and one target box are picked per image, and the threshold rule. If your benchmark used the same settings as the controlled runs, one sentence saying so is enough.

**3. Final cleanup.**

- Delete the draft macros (`\tbd`, `\xx`, `\phfig`) in `main.tex`, so that a forgotten placeholder breaks the build.
- Rebuild, and check that no red text remains.
- Check anonymity: no names of people, Samsung or Kaggle accounts in the PDF or its metadata.
- Optional but cheap: put the code, the evaluation scripts and the per-image outputs in an anonymous repository and cite it. Reviewers can then check every table themselves.
- Commit. Nothing since commit `0cabf9d` is committed yet.

### For a strong accept

**4. Show that the system works for people.** This is the one thing the paper cannot argue its way around. Every result so far measures the model, not the interaction. If the ethics approval allows, it fits before 1 November; otherwise, run it in the revision round. Start the ethics application now either way.

- **Small user study on the glasses:**
  - **Participants:** 12–16.
  - **Task:** point at objects in a cluttered room, placed touching the finger, near it and across the room. The glasses localise the target, a recogniser names the crop, and the name is spoken or shown on the phone.
  - **Conditions,** within subjects and counterbalanced:
    - on-device SpatialNet;
    - streaming frames to the phone or a cloud model, which is more accurate but slower;
    - optional: with and without a confirmation step for large-gap targets, which tests design implication 4.
  - **Measures:** whether the right object was named, time to a correct answer, corrections, NASA-TLX, SUS, preference, and a short interview.
  - **Analysis:** a mixed-effects logistic model for success, with a random effect per participant, and Wilcoxon tests for the questionnaires.
  - **Needs:** ethics approval, the glasses, and about a week.
- **Demo video, 1–2 minutes:** the glasses' view with SpatialNet's outputs running live, with faces blurred, as supplementary material. It is cheap, and reviewers in ubiquitous computing respond to seeing the system work. It is part H of the device request.
- **Optional extras that close limitations the paper states** (all in part H or E of the request):
  - about 30 frames of a non-pointing hand (holding a cup, resting), the first test of the pointing gate;
  - five one-minute clips of natural use: false triggers per minute, with and without smoothing over frames (design implication 5);
  - companion hands (G3): ownership in real scenes;
  - the detectors on the glasses (E): their latency on the same hardware.

## Experiments (14 October 2026)

All runs use the main recipe and seed 42 unless stated, and are scored on the same test images. Brackets give 95% paired-bootstrap intervals; n.s. means not significant.

| Run | What changed | Result | In the paper |
|---|---|---|---|
| E1 | Ray prior acts only in front of the fingertip | P@0.5 +0.3 (n.s.); large gap +5.3 [−0.5, +11.0] (n.s.); large-gap direction 4.3° worse | §7.5, Table 10 |
| E2 | Near and large-gap targets weighted 3× | P@0.5 −1.8, P@0.7 −3.7, overlap −3.8 (all significant); near and large-gap gains n.s. | §7.5, Table 10 |
| E3 | Target-size sample spaced 2 cells | P@0.7 −1.2 and small targets +1.3 (both n.s.) | §7.5, Table 10 |
| Y1 | YOLO26n, ImageNet start, our recipe | 62.9% P@0.5 (your benchmark: 63.2%); SpatialNet is 4.0 lower [−6.0, −2.2] | §7.2, Table 6 |
| Y2 | Y1 plus our distractor pasting | As stable as SpatialNet under pasted hands (+0.7 against +1.2; difference n.s.); pasting gains +3.8 [2.4, 5.2] under interference, no clean cost | §7.2, §7.6, Table 6 |
| Y3 | YOLO26n, COCO start, our recipe | COCO gain +2.4 [0.6, 4.3] on other images, −0.7 (n.s.) on COCO-derived images | §7.2, Table 6 |
| Y4 | Y1 with training seed 1 | 63.6% P@0.5; differs from Y1 by +0.6 [−1.0, +2.2] (n.s.) | §7.2, Table 6 |
| Y1–Y4 | Centre in target, SpatialNet against each detector | +2.8, +3.2, +3.0 and +2.6 points; every interval excludes 0 | §7.2, abstract, introduction |
| R3 | Recognition test: CLIP ViT-B/16 and SigLIP base name the target from each model's crop (2,603 targets, 348 category names) | SpatialNet's crops are named as often as the four detectors' (top-1 differences −1.3 to +0.3; all intervals include 0 and lie within −2.8 to +1.8). On large-gap targets, the detectors' crops are named 4.3–8.1 points more often. | §5.2, §7.2, Table 7, abstract, introduction, §8, conclusion |

The recognition test was first launched on a GPU with larger recognisers (CLIP ViT-L/14 and SigLIP so400m; kernels R0–R2). Those runs stalled in Kaggle's GPU queue or hit a library change, and none produced results. The paper reports R3, which ran the same code on CPU with the base-size recognisers.

**How the claims changed:**

| Topic | Before | Now |
|---|---|---|
| Title | "... with Wearer-Aware Hand Ownership" | Option B title, without ownership |
| Target choice | "Similar" (untested) | SpatialNet picks the labelled object more often than every controlled detector, across two detector seeds |
| What the detectors' tighter boxes are worth | Argued | Measured: nothing for naming the target, except on large-gap targets. Its loose-but-centred boxes are still named correctly 47–51% of the time, against 54–58% for accurate boxes. |
| Robustness to other hands | Credited to the training recipe, untested | Shown: the recipe makes a generic detector equally robust. No longer a SpatialNet-only claim; the recipe is a transferable contribution. Real bystander hands are stated as untested. |
| COCO overlap | "Scores may be optimistic" | No sign of inflation for YOLO26n (NanoDet was not retrained). The 41% overlap stays as a reported fact. |
| Ray prior and large-gap targets | "Hypotheses we did not test" | Tested: no remedy works, which points to the direction estimate as the bottleneck |
| Detector comparison | Point estimates | Paired intervals |

## Changes from the round-3 review (`docs/IMWUT_Review_revised_paper_v2_2026-10-14.md`)

| Finding | What changed |
|---|---|
| R1 On-glasses evidence is thin | Waiting for your device data (step 1). |
| R2 The title promises ownership; real-scene ownership is a placeholder | Option B: new title; §7.6 is synthetic only and says real bystander hands are untested; §7.8 is a wearer-only check on the glasses' frames; section G of the request was rewritten to match. |
| R3 The value of target choice is argued, not shown | Downstream recognition test (§5.2, §7.2, Table 7): SpatialNet's crops serve a recogniser as well as the detectors', except on large-gap targets. |
| m1 Table 5 caption | Now says that none of these detectors was trained with pasted hands, and that Table 6 adds one that was. |
| m2 "We do not claim a more accurate pointing localiser" | Now "We do not claim a more accurate localiser than large models, or tighter boxes than a compact detector." |
| m3 Dense abstract | The three-seed range and the 226M–7B range were removed; one clause on the recognition test was added. |
| m4 "Intended object" | Now "labelled object" everywhere. |
| m5 Tables 5 and 6 overlap | The Table 6 caption says that the ImageNet-start row reproduces Table 5 within 0.3 points. |
| m6 One run per detector | Y4 (second seed) added to Table 6. The two seeds differ by 0.6 P@0.5 (n.s.), and SpatialNet's centre-in-target lead holds against it: +3.2 [1.8, 4.6]. |
| m7 Stale cross-references | §6.2 and §8 now point to the wearer-only check in §7.8. |
| m8 Placeholders | Now only the device numbers, the real-frame check and the benchmark recipe. |
| m9 References | All 11 checked against the publishers' records: MEgATrack's author list was wrong (16 authors, not 6) and MCUNet lacked pages; both are fixed, and DOIs were added. Only the end page of `miller1968response` (277) rests on secondary sources. The new CLIP and SigLIP entries were verified too. |
| Also fixed | The centre-in difference against the COCO-start detector is 2.6, not 2.7: the exact value, 2.6498, had been rounded the wrong way. |

## Reproducing the new results

**Code.** New options, all off by default:

- `model.py`: `roi_forward_only`, `roi_size_step`;
- `loss.py`: `gap_weight`;
- `train.py` and `test_evaluate.py` pass them through.

With default flags, the model's predictions are bit-identical to the committed code.

**Kaggle (all private):**

- datasets `mosharafapurbo/spatialnet-code-exp` and `mosharafapurbo/spatialnet-recognition-inputs`;
- kernels `spatialnet-exp-e1-forward-ray`, `-e2-gap-weight`, `-e3-dilated-size`, `-y1-yolo-imagenet`, `-y2-yolo-imagenet-paste`, `-y3-yolo-coco`, `-y4-yolo-imagenet-seed1` and `-r3-recognition-cpu`. Kernels `-r0-recognition-quick`, `-r1-recognition` and `-r2-recognition` are the failed GPU attempts and can be deleted.

**Notebooks and scripts:**

- the SpatialNet variants in `experiments/E*_*/`, made by `experiments/make_variant.py`;
- the YOLO runs in `experiments/Y*_*/`, made from `experiments/yolo/yolo_experiment_template.py` by `make_yolo_kernels.py`;
- the recognition test in `experiments/recognition/`: `make_inputs.py` builds the uploaded inputs, and `recognition_test.py` is the kernel, copied into `experiments/R3_recognition_cpu/`. On a GPU the same script uses CLIP ViT-L/14 and SigLIP so400m.

**Outputs:** `runs/E1_forward_ray`, …, `runs/Y4_yolo_imagenet_seed1`, `runs/R3_recognition_cpu`.

**Analysis:**

```bash
python paper/scripts/compare_runs.py runs/main_run runs/E1_forward_ray runs/E2_gap_weight runs/E3_dilated_size
python revised_paper/scripts/small_targets.py runs/main_run runs/E1_forward_ray runs/E2_gap_weight runs/E3_dilated_size
python revised_paper/scripts/detector_controls.py runs/main_run runs/Y1_yolo_imagenet runs/Y2_yolo_imagenet_paste runs/Y3_yolo_coco
python revised_paper/scripts/detector_controls.py runs/main_run runs/Y4_yolo_imagenet_seed1 --out=runs/detector_controls_y4.json
python revised_paper/scripts/detector_controls.py runs/Y1_yolo_imagenet runs/Y4_yolo_imagenet_seed1 --out=runs/detector_controls_y1_y4.json
python revised_paper/scripts/recognition_test.py runs/R3_recognition_cpu
```

Each script reproduces the paper's existing numbers before adding new ones. Y4 is scored in separate calls so that the bootstrap intervals of Y1–Y3 stay exactly as before.

**COCO source per image:** `docs/egopoint_sources.csv`, extracted from the raw EgoPoint annotation files.

## COCO overlap: the facts

- The dataset paper says its edited images are COCO photographs, selected through RefCOCO, with a pointing hand added. Each one names its source photograph in the raw annotations.
- In our test split, 842 of the 2,604 images with a target and at least 400 of the 425 no-hand images are COCO train2014 photographs: at least 1,242 images, or 41%.
- 230 test images share their COCO photograph with a training or validation image.
- COCO train2017, on which COCO-pretrained detectors are trained, contains every train2014 image.

## Writing changes compared with `paper/`

- **Title:** option B, without "ownership".
- **Abstract:** one paragraph without bullets.
- **Introduction:** three contributions matched to the evidence; a positioning sentence (an operating point and an analysis, not a better localiser or tighter boxes); sources for the market and battery statements.
- **Related work:** prose; new paragraphs on HCI pointing, on-device hand tracking, tiny models on wearables and compact detectors.
- **Problem definition:** scoped to the two scenes EgoPoint contains; image-plane gap terms (overlap, near, large gap); the latency budget with a citation.
- **Method:** compacted without removing technical content.
- **Results:**
  - the benchmark detector table with the end-to-end score (Table 5), the controlled table (Table 6, with a second detector seed), the recognition test (Table 7) and the remedies table (Table 10);
  - §7.6 covers synthetic other hands only; §7.8 is the new wearer-only check on the glasses' frames;
  - the resolution table folded into Table 5;
  - the backbone table, IoU curves and older qualitative figure moved to Appendix C.
- **Discussion:** six design implications; limitations grouped by construct, internal and external validity.
- **Conclusion:** rewritten to match the claims.
- **Architecture figure (Fig. 2):** the heatmaps are overlaid on the input frame. The same figure is in `paper/` and `submission_paper/`.

## Build

```bash
cd revised_paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
