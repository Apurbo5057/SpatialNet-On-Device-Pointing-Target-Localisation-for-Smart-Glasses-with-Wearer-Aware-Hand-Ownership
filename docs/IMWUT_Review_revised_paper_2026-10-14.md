# IMWUT review of the revised paper (`revised_paper/`)

- **Reviewed:** `revised_paper/main.pdf`, built on 14 October 2026. It had 27 pages, with the body ending on page 22; after the COCO addendum at the end of this file, it has 28 pages, with the body ending on page 23. The references and Appendices A–C follow the body.
- **Compared with:** the first review of `paper/` (`docs/IMWUT_Review_2026-10-14.md`, score 4/10). That file is kept unchanged.
- **Reviewer:** one simulated IMWUT referee, as before. The score is this reviewer's judgement, not a prediction.
- **Assumptions:**
  - Every number in the paper is taken as correct.
  - **T5 is ignored, as you asked.** You will add it tomorrow.
  - **The detector training recipe is pending.** You will provide the benchmark files tomorrow.
  - Both are scored as *pending*, not as missing. Part 4 says how to make each of them count.

---

## 1. Score

**5.5 / 10: major revision.** This is up from 4 / 10.

| Version | Score | Likely outcome |
|---|---|---|
| `paper/`, first review | 4 / 10 | Major revision, leaning reject |
| **`revised_paper/` as it stands** (T5 and recipe pending) | **5.5 / 10** | Major revision |
| + detector recipe + a solid T5 | 6 / 10 | Major revision, clearly worth revising |
| + T5 that also measures target accuracy on the glasses' frames (Part 4, item 2) | 6.5 / 10 | Same, but with the main evidence gap partly closed |
| + a small user task on the glasses (later, in the revision round) | 7+ / 10 | Minor revision becomes possible |

**Why it went up.** The text-level problems are gone:

- the claims now match their evidence;
- the paper positions itself honestly, as an operating point and an analysis rather than a better localiser;
- the comparison with the detectors is framed as a finding, with a clear "when to use which";
- the paper is a third shorter in body pages and much easier to read.

A reviewer can no longer catch it overclaiming.

**Why it is not 6 yet.** What remains is missing *evidence*, not wording:

1. No accuracy is measured on frames from the glasses' own camera.
2. The ownership claim in the title rests on synthetic hands until T5 is in.
3. The detector recipe is missing, so a reviewer cannot judge the paper's strongest comparison.
4. The value of SpatialNet's extra outputs (hand, fingertip, direction) is argued, not shown.

Items 1–3 can be fixed tomorrow (Part 4).

---

## 2. What the revision fixed

| First-review finding | Status | How |
|---|---|---|
| C1 Thin ubicomp contribution | Mitigated | New "Design implications" (§8); claims scoped to execution and cost on the device. Still no user study and no on-device accuracy (see R1). |
| C2 Own baselines undercut the model | Mitigated | Positioning sentence; detectors' lead stated openly and turned into guidance ("What the detectors' lead means", §7.2); contributions reworded. |
| C3 Unfinished text | Partly fixed | `\tbd{confirm}` resolved. T5 and the detector recipe are intentionally left for you. |
| M1 Title claims ownership | Open | Depends on T5 (see R3). |
| M2 Device measurements | Mitigated | Battery reported as "one observation per mode" with its limits; latency budget cited; device scope stated. Protocol details still missing (R5). |
| M3 Four scenes / always-on story | Fixed | §3 scopes the paper to two scenes; §7.3 calls for temporal smoothing; "pointing event" became "hand detected". |
| M4 Overclaiming wording | Fixed | Words like "collapses", "explains", "precisely", "cheaper", "carries over" and "must" are gone or qualified. |
| M5 Cross-protocol parity | Fixed | "Same accuracy range, not parity or superiority"; the re-scoring moved to the table note as an assumption. |
| M6 Fragile ablation inference | Mitigated | Labelled exploratory; supporting effects marked; A4's different training length stated. |
| M7 Incomplete baseline reporting | Open | Recipe and threshold rule pending (R2). |
| M8 Data validity | Mitigated | Worst-case bound for the 43 excluded images; external-validity limitations regrouped. |
| M9 Length and style | Fixed | Body 22 pages (was 29); abstract 240 words (was 425); 9 main tables (was 12) and 6 main figures (was 8); prose instead of nested bullets. |
| M10 Related-work gaps | Fixed | HCI pointing, on-device hand tracking, tiny models on wearables, compact detectors, wearable privacy; 14 references added (R6). |
| Minor m1–m10 | Mostly fixed | Terms are "overlap / near / large gap"; one wording for target choice; "at least 300 times"; keypoint-detector sentence. Open: m8 (the name), m9 (macros, which are still needed). |

---

## 3. Findings on the revised paper

### 3.1 Reviewer's summary

The paper studies how much pointing-target localisation a model small enough for smart glasses can do. SpatialNet is a 0.72M-parameter, image-only CNN that returns the wearer's hand, fingertip, direction and target box in one pass, and it runs in 60–80 ms per frame on a pre-production glasses unit. On EgoPoint it reaches 58.9% target P@0.5, against 63.6–93.9% for large published models and 59.9–67.6% for compact detectors trained by the authors at the same compute. The paper is open that the detectors draw better boxes. It argues that a structured model is preferable when an interface also needs the hand, fingertip and direction. Its ablations, resolution study and probes give design guidance for small pointing models.

### 3.2 Strengths

1. **Honest and well positioned.** Every headline claim now matches its evidence, and the limitations are where a reader expects them.
2. **Rigorous evaluation:** seeds, paired intervals, a seed-spread rule, a duplicate audit, a pretraining control for the detectors, and an end-to-end score.
3. **Useful guidance for practitioners.** The five design rules in §8 are actionable and grounded in the paper's own results.
4. **A real deployment on glasses hardware.**
5. **Compact and readable.**

### 3.3 Remaining findings

| ID | Severity | Finding | Where |
|---|---|---|---|
| R1 | Major | **No accuracy is measured on the glasses' own frames.** The device results cover execution and cost only. The camera is portrait and its frames are squashed to a square, so EgoPoint accuracy may not transfer. This is now the main gap for an IMWUT audience. | §6.2, §7.7, §8 |
| R2 | Major (fix pending) | **The detector recipe is missing.** It should give the classes, epochs, augmentation, the rule that selects one hand box and one target box per image, and the threshold rule for the end-to-end score. Without the selection rule, the pasted-hand comparison cannot be interpreted. | §5.2 |
| R3 | Major (pending) | **The title claims "Wearer-Aware Hand Ownership".** Until T5 is in, ownership rests on a synthetic test with a known positional shortcut. | Title, §7.6 |
| R4 | Major, but can only be argued | **The extra outputs are not shown to matter.** The paper's case for SpatialNet over a detector is its hand, fingertip and direction outputs, and these are argued, not demonstrated. A reviewer may still ask: "why not a detector with a keypoint head?" The paper acknowledges this; only a user or interface study can settle it. | §7.2, §8 |
| R5 | Minor | **Device measurement protocol.** Readers will still ask what the 60–80 ms includes, how many runs or frames it covers, and whether an idle baseline exists. | §6.2, §7.7 |
| R6 | Minor | **Unverified references.** Ten references were added from memory and need checking (listed in `revised_paper/README.md`). A wrong page number or venue is the kind of detail that costs credibility. | `references-revision.bib` |
| R7 | Minor | **Exploratory ablations.** The ablation section calls the per-component reading exploratory. If the target metric for each component was fixed before the runs, saying so would strengthen §7.5 considerably. | §7.5 |
| R8 | Minor | **Untested centre-in-target differences.** They are worded "similar", which is safe. A paired bootstrap on saved per-image detector outputs would let you either keep "similar" or say "higher". This is a computation on existing files, not an experiment. | §7.2 |
| R9 | Minor | **Figure label.** The ablation figure still says "far" while the text says "large gap". The caption explains this, but regenerating the figure from the saved results (plotting only) would make them consistent. | Fig. 4 |
| R10 | Minor | **Figure placement and density.** Fig. 6 floats onto the first Discussion page, and the labels in Fig. 2 are small at print size. | §7.8, §4 |
| R11 | Optional | **Name collision.** "SpatialNet" is also the name of a published speech-separation model (IEEE/ACM TASLP 2024). | Title |

### 3.4 Section by section (abstract, §1–§3, §7)

| Part | Now good | What a reviewer may still say | Your action |
|---|---|---|---|
| Abstract | One paragraph; calibrated; states the detector comparison | "Ownership is only synthetic" | Add the T5 clause (marked by a LaTeX comment at the end of the abstract) |
| §1 Introduction | Clear question; positioning sentence; 3 evidence-matched contributions; sources | "(pasted and real)" only holds once T5 is in | Nothing beyond T5 |
| §2 Related work | Broader HCI and wearable coverage; detectors positioned as the natural alternative | Checks on the new references | Verify the 10 references from memory |
| §3 Problem | Two-scene scope; image-plane gap terms; cited latency budget | — | — |
| §7.1 | Clear "same range, not parity" reading | Cross-protocol caveats (inherent, disclosed) | — |
| §7.2 | Honest comparison and guidance | Recipe missing; no intervals between models | Fill the recipe; optional paired bootstrap (R8) |
| §7.3 | Clear error decomposition; temporal smoothing noted | — | — |
| §7.4 | Hypotheses labelled as such | Single runs (disclosed) | — |
| §7.5 | Exploratory label; supporting effects marked; duplicate check | Pre-designation of the metrics | State it if true (R7) |
| §7.6 | Synthetic test read correctly; recipe confound stated | Real frames | T5 |
| §7.7 | Battery honestly scoped | Latency coverage, run counts, idle baseline | Add the facts you already have (R5) |
| §7.8 | Taxonomy plus complex failures | — | — |

---

## 4. What you still need to do (in priority order)

1. **Fill the detector recipe (§5.2).** Required, about 30 minutes once you have the benchmark files. Write it in one paragraph:
   - classes;
   - input size and epochs;
   - augmentation;
   - pretrained weights;
   - how one hand box and one target box are chosen per image (for example, the highest-confidence box of each class);
   - the validation rule used to tune the hand-gate threshold for the end-to-end score.
2. **T5: make it carry two results, not one.** This is the single biggest lever for reaching 6+.
   - Run the deployed 160 px model on the frames recorded with the glasses.
   - In the same annotation pass, label three things per frame:
     1. Is the predicted hand the wearer's?
     2. Is the fingertip on the wearer's pointing finger?
     3. Does the predicted target box land on the object being pointed at?
   - The second and third labels give the first **accuracy measured on the device's own frames** (a centre-in-target-style rate with a Wilson interval), which answers R1.
   - Report the number of clips, participants and frames; Wilson intervals; Cohen's κ; and at least one failure in Fig. 5.
   - Then update the places that depend on T5:
     - the abstract clause (the LaTeX comment marks the place);
     - the last row of Table 8 and Fig. 5;
     - the consent text in §9;
     - the limitation "the real-frame check is small" (§8), with the actual size;
     - a sentence in §7.7 ("Scope of the device results") if you add target accuracy.
   - **Title decision:**
     - If T5 shows clearly high ownership accuracy on a reasonable number of frames, keep the title.
     - If not, switch to *SpatialNet: Image-Only Pointing-Target Localisation on Smart Glasses with a 0.7M-Parameter CNN*.
3. **Verify the 10 references written from memory** (R6). Check each field against the publisher's page, and fix or remove any you cannot confirm.
4. **Add the device facts you already know** (R5). No new measurement is needed:
   - what the 60–80 ms covers (inference only, or capture plus inference);
   - over how many frames or runs it was observed;
   - whether an idle baseline was recorded, and the starting charge level for the battery minutes.
5. **If the benchmark files include per-image predictions**, run the paired bootstrap for centre-in-target and for the pasted-hand differences (R8). Keep "similar" unless the intervals exclude zero.
6. **Before submitting:**
   - delete the draft macros in `main.tex` and rebuild;
   - check that no red text remains;
   - check anonymity: no account names, vendor names or links in the PDF or its metadata.
7. **Optional, if time allows:**
   - regenerate Fig. 4 with "large gap" labels (plotting script only) (R9);
   - decide on the model name (R11);
   - state whether the ablation metrics were fixed before the runs (R7).

---

## Addendum (14 October 2026): COCO overlap with the test set

**What was checked.** Your hypothesis was that COCO-pretrained detectors had seen EgoPoint test images. It is correct:

- **The dataset paper** says its edited images are COCO photographs, selected through RefCOCO, with a pointing hand added by a multimodal model.
- **The raw annotation files** name each source photograph.
- **In our test split,** 842 of the 2,604 images with a target and at least 400 of the 425 no-hand images are COCO train2014 photographs: at least 41% of the test set.
- **NanoDet-Plus-m and the COCO-pretrained YOLO26n** are trained on COCO train2017, which contains every train2014 image.
- **SpatialNet** (ImageNet backbone), **YOLO26n with an ImageNet start** and **YOLO26n from scratch** never saw COCO data.

**What changed in `revised_paper/`:**

- **§5.1:** source counts, and 230 test images that share their COCO photograph with training or validation images.
- **§5.2:** a "Pretraining overlap" paragraph.
- **Table 5:** split into "no COCO data" and "COCO-pretrained (†)" groups.
- **§7.2:** leads with the like-for-like ImageNet comparison (4.3 points); the COCO effect for YOLO26n is bounded at 1.8 points; NanoDet's lead is stated as unsplittable.
- **Abstract, introduction, discussion and conclusion:** updated to match. The abstract now quotes 63.2% for the like-for-like detector instead of the 59.9–67.6% range.

**Effect on the review.**

- **C2 (the baselines undercut the model) is weaker.** The fair reference is now explicit, and the paper reports a dataset fact that matters to anyone benchmarking on EgoPoint, which reviewers tend to value.
- **The score moves from 5.5 to about 5.75 / 10.** The like-for-like detector still leads by 4.3 points, so the main finding is unchanged.
- **The path to 6+ is still T5 and the recipe.**

**Do not overclaim this.** The paper must not say that NanoDet's lead *comes from* the overlap. No NanoDet run without COCO exists, and COCO pretraining also brings genuine transfer: the targets are mostly COCO categories, such as person, motorcycle and chair.

**Add to tomorrow's list (item 5), once the per-image detector outputs arrive.** Score every model separately on the 842 COCO-derived and the 1,762 other test images with a target. Then check:

- whether YOLO26n-COCO's gain over YOLO26n-ImageNet is concentrated on the COCO-derived images;
- whether the same holds for NanoDet-Plus-m against SpatialNet.

If it is, you can state the overlap effect with numbers. This is a computation on saved predictions, not an experiment.

**Only in the revision round, because each needs device or participant time:**

- an idle-baseline battery measurement;
- the latency of a compact detector on the glasses;
- a small user task, such as pointing at objects with and without ray feedback. This is what would move the paper to 7+.
