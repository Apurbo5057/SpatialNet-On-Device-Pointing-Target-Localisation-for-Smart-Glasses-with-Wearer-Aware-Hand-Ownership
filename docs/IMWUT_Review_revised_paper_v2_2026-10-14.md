# IMWUT review of the revised paper, round 3 (`revised_paper/`, with the new experiments)

- **Reviewed:** `revised_paper/main.pdf`, 30 pages. The body ends on page 25, followed by the references and Appendices A–C.
- **Reviewer:** one simulated IMWUT referee, using the same criteria as the first two reviews:
  - `docs/IMWUT_Review_2026-10-14.md`: 4/10;
  - `docs/IMWUT_Review_revised_paper_2026-10-14.md`: 5.5/10, raised to about 5.75 after the COCO check.
- **Scope:** the paper only. All numbers are taken as correct.
- **Pending items, not counted against the paper:** T5, the Samsung device data, and the recipe of the original benchmark detectors. Each is listed below with what it would add.

---

## 1. Score

**6 / 10: major revision, clearly on track for acceptance.**

| Version | Score |
|---|---|
| Original `paper/` | 4 |
| `revised_paper/`, writing only | 5.5 |
| `revised_paper/` with the six new experiments (this review) | **6** |
| + T5 real frames and repeated device measurements from Samsung | 6.5–7 |
| + a downstream test of what target choice buys (section 4, item 2) | about 7 |
| + a small user study on the glasses | 7.5+ |

**Why it went up.** The two findings a reviewer would have attacked are now tested:

- the "untested hypotheses" are gone: three remedies were tried, none works, and that is reported;
- the detector comparison now has controlled runs with paired intervals;
- the robustness claim is attributed correctly, to the training recipe, which also works for a generic detector;
- the COCO concern became a measured result rather than a caveat;
- SpatialNet now has a tested advantage: it picks the labelled object more often than every controlled detector.

The paper is honest, coherent and hard to fault on rigour.

**Why it is not higher yet.** The IMWUT-specific evidence is still thin:

- the on-glasses numbers are single observations;
- ownership in real scenes is still a placeholder;
- the value of SpatialNet's advantage (target choice, extra outputs) for an interface is argued, not shown.

None of this is a writing problem; it is the data you are already collecting, plus one optional experiment.

---

## 2. Summary of the submission (reviewer's view)

SpatialNet is a 0.72M-parameter, image-only CNN that runs on pre-production smart glasses in 60–80 ms. In one pass it predicts:

- the wearer's hand;
- the fingertip;
- the pointing direction;
- the box of the pointed-at object.

On EgoPoint it reaches 58.9% target P@0.5. Compact detectors at the same compute draw tighter boxes (+4.0 P@0.5 and +11.6 P@0.7 for the like-for-like YOLO26n), but SpatialNet places its target centre on the labelled object more often (+2.7 to +3.0 points against three controlled detectors).

The paper also shows:

- training with pasted distractor hands makes any of these models robust to synthetic other hands;
- 41% of the EgoPoint test images are COCO photographs, but COCO pretraining does not exploit this;
- resolution is the largest accuracy lever;
- the learned ray prior encodes proximity, and neither a forward constraint, re-weighting large-gap targets, nor a wider size sample fixes large-gap or box-extent errors.

---

## 3. Strengths

1. **Exemplary experimental honesty.**
   - Negative results are reported and explained: the remedies, the ray branch, and the detectors' box quality.
   - Every comparison states its protocol.
   - Effects are counted only if they beat both a paired interval and the retraining spread.
2. **Controlled baselines.** Detectors are retrained under one recipe with the same images and scoring code, and their recipe reproduces the earlier benchmark (62.9% against 63.2%).
3. **A transferable contribution.** The distractor-pasting recipe works beyond SpatialNet: it makes a generic detector equally robust at no clean cost. That makes the paper useful to anyone building pointing models.
4. **Benchmark hygiene.** The COCO-overlap analysis (41% of the test images; no measurable inflation for YOLO26n) is a useful contribution for every future EgoPoint paper.
5. **A clear trade-off.** The trade-off with detectors (target choice versus box extent) is stated, quantified and turned into design rules.
6. **Real deployment** on glasses hardware, with an honest statement of what was and was not measured there.

---

## 4. Remaining findings

### Major

| ID | Finding | What fixes it |
|---|---|---|
| R1 | **On-glasses evidence is thin.** Latency is a range without a breakdown; battery is one minute per mode; accuracy is never measured on device frames. For IMWUT this is still the main weakness. | The Samsung data: sections A–D of the request, then rewriting §6.2, §7.7 and Table 11. Section A (the deployment check) lets you say that the on-glasses model computes the same outputs as the evaluated one. |
| R2 | **Ownership in real scenes is a placeholder**, yet the title promises "Wearer-Aware Hand Ownership". The paper now shows that this robustness comes from the training recipe, so the title rests on the recipe and on T5. | T5 (Samsung section G). If T5 is weak, use a title without "ownership", e.g. *SpatialNet: Image-Only Pointing-Target Localisation on Smart Glasses with a 0.7M-Parameter CNN*. |
| R3 | **The value of SpatialNet's advantages is argued, not shown.** The paper's case is that choosing the right object (+2.7–3.0 points) and returning the hand, fingertip and direction matter more for a pointing interface than a tighter box (−11.6 at P@0.7). A reviewer will ask: *does a 3-point target-choice gain survive the 12-point box-precision loss in anything a user experiences?* | Best without people: a **downstream recognition test**. Crop each model's predicted target box, run an off-the-shelf zero-shot classifier (for example CLIP) on the crop, and score whether it names the labelled target category (EgoPoint has the category names). This is a few hours on Kaggle with no new training, and it measures directly which trade-off a recogniser prefers. Otherwise, keep the current hedged wording, which is defensible. |

### Minor

| ID | Finding | Fix |
|---|---|---|
| m1 | Table 5's caption says "only SpatialNet was trained with pasted hands". That is true of Table 5's rows, but it now reads as if no detector was ever trained with pasting. | "None of these detectors was trained with pasted hands; Table 6 adds one that was." |
| m2 | The introduction says "We do not claim a more accurate pointing localiser", then claims better target choice. | "We do not claim tighter boxes than a detector." |
| m3 | The abstract (about 270 words) is dense with numbers. | Optional: drop the three-seed range and the 226M–7B range; both are in the body. |
| m4 | "Intended object" in the abstract and §7.2 is interpretive; the metric is the *labelled* object. | Say "labelled object". |
| m5 | Two detector tables (5 and 6) contain the same configuration with slightly different numbers (63.2% and 62.9%). | Already explained in the text. Optionally, one sentence in the Table 6 caption: "the ImageNet-start row reproduces Table 5 within 0.3 points". |
| m6 | One training run per controlled detector. The centre-in-target result holds across three different detector trainings, which partly answers this. | Y4 (a second seed), if it finishes in time; otherwise say "three detector trainings" in the text. |
| m7 | §8 still says "the real-frame check is small", and §6.2 still says real frames are "assessed in Section 7.6". | Update both once T5 is in. |
| m8 | Placeholders: T5 (§7.6, Table 10, Fig. 5, §9, the abstract comment) and the benchmark recipe (§5.2). | Fill them in, then delete the draft macros. |
| m9 | 11 references were written from memory. | Verify every field before submitting. |

### Resolved since the last review

- C2: own baselines undercut the model. Now a stated trade-off, with a tested advantage.
- M3: four-scene story.
- M4: overclaiming.
- M5: cross-protocol parity.
- M6: untested ablation hypotheses, now tested.
- M7: baseline intervals.
- M9: length and style.
- M10: related work.
- The COCO question.
- The robustness confound.

---

## 5. What would raise the score, in order of value per hour

1. **Samsung sections A–D:** deployment check, latency breakdown, memory, repeated power with an idle baseline. Then rewrite §6.2, §7.7 and Table 11. (+0.5)
2. **T5 (Samsung section G).** Fill in §7.6, Table 10, Fig. 5, §9 and the abstract clause, and decide on the title. (+0.5, and it protects the title)
3. **Optional downstream recognition test** (R3): a few hours on Kaggle, no new training. It turns "what the detectors' lead means" from an argument into evidence. (+0.5)
4. **Minor fixes m1–m9:** an hour of writing.

With items 1, 2 and 4, I would expect reviewers to move from "major revision" to "minor revision". A small user study remains the step to a clear accept, and it fits the revision round.
