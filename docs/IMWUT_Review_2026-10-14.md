# IMWUT review of the SpatialNet paper (`paper/`)

- **Reviewed:** `paper/main.pdf` at commit `0cabf9d` (14 October 2026). Title: *SpatialNet: Compact Pointing-Target Localisation for Smart Glasses with Wearer-Aware Hand Ownership*. 32 pages: body to p. 29, then references and two appendices.
- **Reviewer:** one simulated IMWUT referee. The score is this reviewer's judgement, not a prediction of what the panel will decide.
- **Assumptions:** every number in the paper is taken as correct. The review checks whether the claims follow from the numbers, not the arithmetic. Nothing in `paper/` or the code was changed.

**How this report is organised**

- **Part 1:** the review as a referee would submit it: summary, strengths, findings and score.
- **Part 2:** detailed findings for the abstract, Sections 1–3 and Section 7 (Results).
- **Part 3:** how to fix each finding by rewriting, cutting or moving text, without new experiments.

**Severity labels**

- **Critical:** can sink the paper on its own.
- **Major:** a referee will require it in a revision.
- **Minor:** polish.

**Where the line numbers point** (files at commit `0cabf9d`):

| Paper part | File |
|---|---|
| Title, abstract | `paper/main.tex` |
| §1 Introduction | `paper/sections/01-introduction.tex` |
| §2 Related Work | `paper/sections/02-related-work.tex` |
| §3 Problem Definition | `paper/sections/03-problem.tex` |
| §4 Method | `paper/sections/04-method.tex` |
| §5 Data and Protocol | `paper/sections/05-data-protocol.tex` |
| §6 Implementation and Deployment | `paper/sections/06-implementation.tex` |
| §7 Results | `paper/sections/07-results.tex` |
| §8 Discussion | `paper/sections/09-discussion.tex` |
| §9 Ethics, §10 Conclusion | `paper/sections/10-ethics-conclusion.tex` |

---

## Part 1: The review

### 1.1 Summary of the submission

The paper asks how much pointing-target localisation a model small enough for smart glasses can do. SpatialNet is a 0.72M-parameter CNN. From one RGB frame, in one pass, it predicts:

- the wearer's hand box;
- the fingertip;
- the image-plane pointing direction;
- the box of the pointed-at object.

Its heads are chained: each one reads the shared feature map where the previous head predicted.

On the EgoPoint test split, it reaches 92.6% hand P@0.5 and 58.9% target P@0.5, falling to 25.4% for targets far from the fingertip. Two groups of models are more accurate: large published models (226M–7B parameters), and compact detectors that the authors trained at the same compute (59.9–67.6%). Ablations, a resolution study and probes analyse which components matter. The 160 px model runs in 60–80 ms per frame on a pre-production glasses unit.

### 1.2 Strengths

1. **Real deployment.** The model runs on glasses hardware. Few pointing-grounding papers report anything on a wearable.
2. **Careful, transparent evaluation**, well above the norm:
   - three training seeds;
   - paired bootstrap intervals, and a seed-spread rule for ablations;
   - results by fingertip–target gap, and an oracle split of position errors versus extent errors;
   - a duplicate audit;
   - matched-compute baselines with a pretraining control;
   - an end-to-end score that counts gate errors.
3. **Candour.** The paper reports results that do not favour it:
   - detectors are more accurate;
   - the ray branch does not help;
   - two effects weaken once duplicates are removed.

   It also states every cross-protocol caveat.
4. **Useful diagnostic findings.** Resolution is the largest lever at no parameter cost. The learned ray prior mostly encodes proximity. A bigger backbone does not help. Anyone building small pointing models can use these.
5. **Clear problem and motivation**, with a privacy argument for keeping the processing on the device.

### 1.3 Findings at a glance

**Critical**

| ID | Finding | Where |
|---|---|---|
| C1 | **Thin ubicomp contribution.** No user study, no in-situ test, no accuracy measured on the glasses. The on-device evidence is one latency range, one memory figure and one battery minute per mode. IMWUT reviewers will ask what the wearable community learns beyond a vision benchmark. | §1, §6.2, §7.7, §8 |
| C2 | **The paper's own baselines undercut the model's value.** At the same compute, generic detectors trained on the same data are more accurate (+1.0 to +8.7 P@0.5), and they are also image-only and single-pass. Each advantage claimed for SpatialNet is weak: centre-in-target is small and untested, pasted-hand stability is confounded, and the fingertip and direction outputs are unused. | Abstract, §1, §7.2 |
| C3 | **Unfinished text (critical, but easy to fix).** Red placeholders remain: T5 (paragraph in §7.6, last row of Table 11, Fig. 6, consent text in §9), the detector recipe in §5.2, and "confirm" in §8. Several sentences also point to a real-frame check that does not exist yet. A submission with red placeholders risks being returned unreviewed, or being reviewed as incomplete. | §1, §5.2, §6.2, §7.6, §8, §9 |

**Major**

| ID | Finding | Where |
|---|---|---|
| M1 | **Title promises more than the evidence.** It promises "Wearer-Aware Hand Ownership", but ownership is shown only on synthetic pasted hands, and the paper itself shows that test has a positional shortcut. The introduction even calls distractor pasting "a supporting result". | Title, §1, §7.6 |
| M2 | **On-device measurements are under-specified.** Latency is a range, with no trial count or breakdown. Battery is one minute per mode at 1% resolution, with no idle baseline reported. With the stated 160 mAh battery, readers will compute about 11 minutes from full to empty (≈0.9 A average). The general claim "on-device processing is cheaper than offloading" rests on this. | §6.2, §7.7; repeated in the abstract, §1, §8, §10 |
| M3 | **The four-scene problem and the "always-on trigger" story outrun the data.** EgoPoint has no non-pointing hands and no pointing-without-target, so the pointing and target gates never see a negative example. The hand gate raises false alarms on 6.6% of empty frames. | §1, §3, §4.4, §7.3, §8 |
| M4 | **Wording overstates the evidence.** Phrases such as "localise precisely", "keep its boxes precise", "collapses", "explains", "carries over to other pointing models", "must be constrained", "cheaper than offloading" and "parity" rest on single runs, observational probes or confounded comparisons. | Abstract, §1, §7.4–7.7, §8, §10 |
| M5 | **Comparisons with large published models are cross-protocol.** The image sets differ, IoU ≥ versus > differs, and the scoring of no-target images is unknown. The "parity" claim relies on a hypothetical re-scoring. | Abstract, §1, §7.1 |
| M6 | **Ablation inference is fragile.** One seed per variant, many metrics with no multiplicity control, and per-component metrics that look chosen after the fact. The interventions are bundled, the best epochs range from 54 to 120, and two P@0.7 effects lose significance without duplicates. | §7.5 |
| M7 | **Baseline reporting is incomplete.** Missing: the detectors' training recipe and box-selection rule, the rule behind the validation-tuned thresholds, and intervals for SpatialNet-versus-detector differences. Without the selection rule, the pasted-hand comparison cannot be interpreted. | §5.2, §7.2 |
| M8 | **Data validity limits generalisation.** 61% of EgoPoint is composed or generated, and 4.7% of test images duplicate training or validation images. 43 "unclear" test positives are excluded, and frames are squashed to a square. The device camera is portrait, and accuracy is never measured on its frames. | §5, §6.2, §7 |
| M9 | **Length and style.** About 29 body pages and a 425-word abstract with bullets. There are nested bullet lists in most sections, and the headline numbers appear in five places. The main text has 12 tables and 8 figures. | Whole paper |
| M10 | **Related work misses literatures IMWUT reviewers expect.** These are: pointing and selection in AR and head-worn interfaces, egocentric gesture recognition and on-device hand tracking, tiny models on wearables, compact detectors (now the main baseline), and privacy around wearable cameras. | §2 |

The ten minor findings (m1–m10) are listed in Part 2.7.

### 1.4 Score

**4 / 10: major revision, leaning reject, as the paper stands.**

Scale used:

| Score | Meaning |
|---|---|
| 1–2 | Clear reject |
| 3–4 | Below the bar, but revisable |
| 5–6 | Major revision |
| 7–8 | Minor revision |
| 9–10 | Accept |

- **Why not lower:** the evaluation is rigorous and honest, the model really runs on glasses, and the negative results are useful.
- **Why not higher:**
  - the IMWUT-specific contribution (interaction, wearable system, real-world use) is thin (C1);
  - the paper's own baselines leave the model without a clear advantage (C2);
  - several headline claims go beyond their evidence (M1–M5).
- **What would raise it:**
  - The text-only changes in Part 3 should bring it to about **5–6 / 10**: a clean major revision, which is the realistic goal for the first round.
  - Reaching 7 or more needs evidence that wording cannot supply: the real multi-person check (T5) and some accuracy measured on frames from the glasses, ideally with a small user task.

### 1.5 Judgement per review dimension

| Dimension | Judgement | Reason |
|---|---|---|
| Originality | Partly meets | Known components, combined; the novelty is the on-glasses operating point and the analysis |
| Methodological rigour | Meets for the benchmark; partly meets for the device | Strong statistics on EgoPoint; the device measurements are single and coarse |
| Evidence for the claims | Partly meets | The ownership, battery and "precision" claims exceed their evidence; no accuracy on the device |
| Coherence of the argument | Partly meets | The title and contributions promise more than §7.2 and §7.6 deliver |
| Writing | Partly meets | Precise, but long, full of lists and repetitive; placeholders remain |
| Use of the literature | Partly meets | Pointing grounding is well covered; HCI, wearable and privacy literature is thin |
| Significance for IMWUT | Partly meets | On-glasses feasibility matters; its value for interaction is argued, not shown |

---

## Part 2: Detailed findings

### 2.1 Abstract (`main.tex`, l. 40–50)

- **A1 (M9): too long, and in a format that will not survive.** About 425 words, four paragraphs and a bullet list. IMWUT abstracts are typically a single paragraph; aim for about 200–250 words. The abstract pasted into the submission system and into the ACM metadata is plain text, so the bullets will be flattened.
- **A2 (M4, C2): "To make this possible at that size, its heads form an instance-anchored chain" (l. 42).** This says anchoring is what makes the result possible. But ablation A4 shows P@0.5 does not change without it, and §7.2 shows that detectors with no anchoring do better.
- **A3 (M5): "comparable to zero-shot 7–8B multimodal LLMs reported on the same split" (l. 44).** There is no protocol caveat. "Same split" is also not exact: the paper excludes 43 test positives and all images without a target.
- **A4 (C2): "places the box centre on the right object slightly more often (84.8% against 81.6–82.9%)" (l. 44).** This uses a metric the paper introduces, the differences are 2–3 points, there is no significance test, and elsewhere the paper says "at least as often". It reads as a favourable metric chosen after the fact.
- **A5 (M1, C2): "the only model whose accuracy does not drop when other hands are pasted into view" (l. 44).** True, but only SpatialNet was trained with pasted hands, and §7.2 itself says the stability comes from the training recipe. Without that clause the sentence misleads.
- **A6 (M4): "collapses into a fingertip-proximity prior" (l. 48).** The probe gives ρ = −0.38 for proximity and −0.27 for the ray line. Proximity dominates, but nothing collapsed.
- **A7 (M2): "running it continuously drains less battery than merely streaming frames to a phone" (l. 50).** One minute per mode is presented as a general result.
- **A8 (M1): title against abstract.** The title leads with hand ownership, but the abstract mentions it only through the pasted-hands sentence.

### 2.2 Section 1: Introduction

- **I1 (M10): statements without sources.** Two statements need sources: "Camera-equipped smart glasses are becoming everyday devices" (l. 3) and "a battery of a few hundred milliampere-hours" (l. 9).
- **I2 (M4): "One design choice lets such a small network localise precisely: instance-anchored heads" (l. 26).** The paper contradicts this:
  - P@0.7 is 31.8%;
  - detectors draw tighter boxes (§7.2);
  - anchoring's P@0.7 effect does not survive the duplicate check.

  What anchoring does show is a more precise hand box and direction than a plain model of the same size.
- **I3 (M3): "with presence gates for each" (l. 24).** Two of the three gates cannot be trained or tested on EgoPoint (§4.4 says so).
- **I4 (C3): "behaviour when other people's hands are in view (pasted and real)" (l. 36).** The real-frame part is not in the paper.
- **I5 (M5): "comparable to zero-shot ... close to a 7B model fine-tuned on EgoPoint (63.6%)" (l. 42).** "Close" describes a 4.7-point gap measured under different protocols.
- **I6 (M9): repetition.** "Results in brief" (l. 39–46) repeats the abstract, and the contributions repeat it again.
- **I7 (C2): contribution 1 (l. 51–55).** "Image-only, single-pass" no longer sets SpatialNet apart, because the detector baselines are image-only and single-pass too. Only "runs on smart glasses" and the multi-output design remain distinctive.
- **I8 (M4, M6): contribution 3 (l. 64–69).** Two problems:
  - "Distractor pasting makes the model robust to other hands" is shown only on synthetic hands.
  - "This explains why far targets stay hard, and indicates how future models should constrain the ray" turns an observational probe into a causal explanation and an untested prescription.
- **I9 (C1): contribution 4 (l. 70).** "An evaluation that places accuracy next to cost" is not a contribution by itself at IMWUT. It becomes one if framed as a characterisation of the design space that others can reuse.
- **I10 (M1): l. 72 against the title.** The introduction demotes distractor pasting to "a supporting result", while the title promotes ownership.
- **I11 (M9): style.** Nine bullet lists (counting nested ones) in under three pages make the introduction read like notes rather than an argument.

### 2.3 Section 2: Related Work

- **R1 (M10): missing literatures.** A ubicomp referee will expect these. Candidates to check (verify each before citing):
  - **Pointing and selection in AR/VR and head-worn interfaces:**
    - Argelaguet and Andujar, survey of 3D object selection techniques (Computers & Graphics, 2013);
    - Mayer et al., offset correction and cursor feedback for mid-air pointing (CHI 2018).
  - **Interaction on smart glasses:** Lee and Hui, survey of interaction methods for smart glasses (IEEE Access, 2018).
  - **Egocentric gestures and on-device hand tracking:** EgoGesture (IEEE TMM, 2018), MediaPipe Hands (arXiv, 2020), MEgATrack (ACM TOG, 2020).
  - **Tiny models on wearables:** MCUNet (NeurIPS 2020), TinyissimoYOLO (ECCV 2024 Workshops).
  - **Wearable-camera privacy, for §9:**
    - Denning et al., bystanders of AR glasses (CHI 2014);
    - Hoyle et al., privacy behaviours of lifeloggers (UbiComp 2014).
- **R2 (C2): compact detectors are missing.** They are now the main baseline, yet they do not appear in related work. The paper never says why a structured model should beat a fine-tuned detector, and that is the first thing §7.2 tests.
- **R3 (M9): lists instead of synthesis.** The pointing-grounding bullets (l. 8–14) and "What SpatialNet changes" (l. 15–22) repeat the introduction. Prose that contrasts the approaches would read better.
- **R4 (good):** "Explicit ray reasoning is thus not new" (l. 15) is exactly the kind of honest positioning reviewers like. Keep it.

### 2.4 Section 3: Problem Definition

- **P1 (M3): the four-scene formulation (l. 11) promises states that the data cannot provide.** Scenes 100 and 110 never occur in EgoPoint, so the pointing and target gates are trained only on positives. §4.4 says this, but §3 is where a reviewer forms expectations.
- **P2 (M3): "only when a pointing event occurs" (l. 19).** The model has no event detection: it works on single frames, and its pointing gate never sees a negative.
- **P3: the budget is not testable as stated (l. 16).** "Within an interactive latency" needs a number and a citation.
- **P4: terms used later are not defined.** "Touched/on object", "near" and "far" are image-plane categories, not physical contact or depth. Defining them in §3 stops a reviewer from reading "far" as distance from the wearer.
- **P5 (good):** the scope paragraph, which puts target recognition out of scope, is clear.

### 2.5 Section 7: Results

**General**

- **G1 (M9): size.** §7 runs from page 14 to 26, with 9 tables and 5 figures. Readers lose the thread. The core story is buried in sub-bullets: operating point, then detectors, then errors, then what matters, then the device.
- **G2 (M4): headlines stronger than caveats.** Most numbers carry careful caveats, but the sentence above each caveat claims more than it allows. A reviewer quotes the headline, not the caveat.

**7.1 Pointed-target localisation**

- **7.1-a (M5): "We therefore claim parity at a fraction of the size" (l. 84).** This follows five protocol caveats. The 57.8% figure (l. 79) is a hypothetical re-scoring, and presenting it as evidence of parity invites objection.
- **7.1-b (minor): VistaRef's 22.9 images/s (l. 86).** That was measured on a server GPU. It cannot be compared with on-glasses latency and adds nothing.
- **7.1-c (good): the simple references.** A mean-size box at the true fingertip reaches only 11.4%, which shows that the target head does real work. Keep them.

**7.2 Compact detectors at matched compute**

- **7.2-a (C2): the conclusion is left unanswered.** The section is honest, but it ends with "The detectors are therefore the stronger choice when only the target box matters" (l. 151). That hands a reviewer the argument for rejection unless the paper says clearly what the extra outputs are for.
- **7.2-b (M7): the recipe placeholder (§5.2, l. 71).** This is the first thing a reviewer will ask about: classes, epochs, augmentation, and how one hand box and one target box are chosen per image. Without the selection rule, the pasted-hand comparison cannot be interpreted, because a detector might pick a pasted hand.
- **7.2-c (C2, M6): "SpatialNet picks the right object at least as often" (l. 145).** The differences are 2–3 points, untested, on a metric this paper introduces. With 2,604 images, an unpaired 95% interval for the difference between two such rates is roughly ±2 points. The NanoDet comparison (1.9 points) is therefore within noise.
- **7.2-d (M1): "Only SpatialNet's accuracy does not drop" (l. 146).** The bullet below it already says this comes from the training recipe. The headline should say so too.
- **7.2-e (M7): no threshold rule.** The rule behind the validation-tuned thresholds in Table 6 is not stated.
- **7.2-f (good): the pretraining control** (COCO, ImageNet, scratch) is exactly what a careful reviewer would ask for.

**7.3 Per-head accuracy and target distance**

- **7.3-a (M3): false alarms.** The hand gate raises false alarms on 6.6% of empty frames (l. 186). At 12–17 frames/s and without temporal smoothing, an always-on loop would therefore trigger often when no hand is present. This bears directly on the "always-on front end" claim in §8 and should be said here.
- **7.3-b (minor): "touched" (l. 191–192).** This means image-plane overlap, not contact.
- **7.3-c (good): the oracle analysis.** True size gives 70.4% and true centre gives 72.3%, a clear and useful split of the errors.

**7.4 Input resolution**

- **7.4-a (M4): "Beyond 224 px, far targets are limited by the direction estimate, not by the input" (l. 229).** This is a causal reading of two single runs (35.9% against 34.9%) plus a flat direction error.
- **7.4-b (M4): more confidence than one run per resolution supports.** "Their gains are, if anything, underestimated" (l. 231) and "a steady 2.7–3.5 points" (l. 226) both overstate.
- **7.4-c (good):** the confound at 320 px (the data were re-converted) is disclosed.

**7.5 Ablations**

- **7.5-a (M6): "read each one first on the capability its component is meant to provide" (l. 237).** Unless those capability metrics were fixed before the runs, a reviewer will see metrics selected after the fact from about 40 comparisons (Table 9 and Fig. 5).
- **7.5-b (M4): "The anchored chain improves everything that requires precision" (l. 278).** The P@0.7 effect disappears without duplicates, and detectors without any anchoring are more precise still.
- **7.5-c (M6): "Shorter training could widen its gaps, but it would be expected to lower P@0.5 as well" (l. 285).** This is speculation used as evidence.
- **7.5-d (M4): one model generalised to all pointing models.** Three sentences do this and prescribe a fix that was not tested:
  - "Probing the trained full model explains this" (l. 304);
  - "the training signal rewards a 'near the fingertip' prior" (l. 310);
  - "The finding carries over to other pointing models ... The ray must be constrained" (l. 314).
- **7.5-e (minor): mixed samples in the A3 probe.** It mixes a 500–800-image subset with whole-split numbers in one list, and readers will ask why.
- **7.5-f (good):** the robustness bullets (l. 316–323) and the duplicate footnote in Table 9 are exemplary.

**7.6 Following the wearer's hand**

- **7.6-a (C3): placeholders.** The "Real frames" paragraph, the last row of Table 11 and Fig. 6 are all placeholders.
- **7.6-b (M1): the title's claim rests on small numbers.** The section itself shows the ownership metric is near its ceiling because of a positional shortcut (93% already in the first epoch). That is honest, but it leaves the title's claim resting on two smaller numbers: the change in no-hand frames kept, and the A1 contrast.
- **7.6-c (good):** "Against compact detectors" (l. 358) correctly credits the effect to the training recipe.

**7.7 On-device performance**

- **7.7-a (M2): latency.** A range (60–80 ms) is given without:
  - the number of trials;
  - a distribution;
  - what the time includes (capture, resize, inference, decoding).

  The "100–200 ms budget" (l. 421) has no citation.
- **7.7-b (M2): battery.** "On-device processing is cheaper than offloading" (l. 423) is a general claim from one minute per mode. The paper reports no idle baseline, so the share due to SpatialNet is unknown. The current implied by the stated 160 mAh battery (about 0.9 A and 1.5 A) is high for a 45 g frame, so expect a reviewer to ask what else was running.
- **7.7-c (M4): "The comparison favours offloading, and on-device still wins" (l. 424).** This is rhetoric, not evidence.
- **7.7-d (C1, M8): no accuracy on the device.** Nothing in this section shows the model's accuracy on the device's own frames. Those frames are portrait and are squashed to a square (§6.2).

**7.8 Qualitative results**

- **7.8-a (minor): overlapping figures.** Figures 7 and 8 overlap. One figure with successes and labelled failure types would do.
- **7.8-b (good):** the captions say the examples are deliberately selected, and the rule-based failure taxonomy is clear.

### 2.6 Outside the detailed scope (brief, because the same claims recur)

- **§5.1, l. 10:** "so that our numbers are directly comparable with theirs" contradicts the protocol caveats in §7.1.
- **§5.2, l. 62 and §6.2, l. 49:** both point to a real-frame check in §7.6 that does not exist yet.
- **§6.1, l. 11:** "a completed schedule would, if anything, raise the absolute numbers" is speculation.
- **§4.4, l. 113:** "another person's pointing hand ... cannot switch them on by itself" is an untested design claim; "is designed so that" is safer.
- **§8:** "a good always-on front end" (l. 8), `\tbd{confirm}` (l. 19), "the real-frame check ... is small" (l. 28) and "Our measurements support this" (l. 47).
- **§10, l. 16:** the conclusion repeats "keep its boxes precise", "collapses", "follow the wearer's hand" and "costs less battery".

### 2.7 Minor findings

- **m1:** "Touched", "near" and "far" are image-plane categories. Call them "overlap", "near" and "large gap", or add "in the image" each time.
- **m2:** cite the 100–200 ms interactive budget, for example Miller 1968; Card, Robertson and Mackinlay 1991; or Nielsen 1993 (verify before citing).
- **m3:** "300 times smaller than VistaRef" uses the Base backbone (226M), but the cited accuracy is from the Large backbone. Write "at least 300 times".
- **m4:** the same result is worded "slightly more often" (abstract) and "at least as often" (§1, §7.2, §10). Use one wording.
- **m5:** Fig. 2's labels are very small at print size, and Table 5 squeezes 13 columns into 2.1 pt spacing.
- **m6:** the hand-gate threshold is 0.5 in §7.3 but tuned on validation in Table 6. Explain both in one sentence.
- **m7:** "Centre in target" is defined in §5.2 as a diagnostic, but it is used as a headline result.
- **m8:** the name "SpatialNet" is already used by a published multichannel speech-separation model (IEEE/ACM TASLP, 2024). A reviewer may mention it. Renaming is optional.
- **m9:** delete the draft macros (`\tbd`, `\xx`, `\phfig`) before submitting, as `paper/README.md` already says, so that a forgotten placeholder breaks the build.
- **m10:** a detector with a keypoint head (for example a pose variant of YOLO) is an obvious alternative for fingertip output. One sentence in the limitations pre-empts the question.

---

## Part 3: Solutions without new experiments

### 3.1 Ground rules

- **Disclosed weaknesses cost little; discovered ones cost a lot.**
  - Reviewers forgive limitations that the authors state plainly and in proportion.
  - They punish claims that turn out to be larger than the evidence.
  - So the safest paper states every weakness first, in your own words, and makes each headline claim no stronger than its weakest caveat.
- **Do not remove unfavourable results.**
  - The detector comparison, the duplicate audit and the cross-protocol caveats are what make this paper credible.
  - Any reviewer who knows YOLO or EgoPoint would notice their absence.
  - Removing them would turn honest framing into misreporting.
- **What writing can and cannot fix.**
  - Rewriting fixes M1, M3–M7, M9 and M10, and most of C2 and C3.
  - It cannot add a user study, real multi-person frames or accuracy on device frames (C1).
  - Aim for a clean major revision now, and add T5 in the revision round.

### 3.2 One positioning statement, used everywhere

State this idea, in these terms, in four places: the abstract, the end of the introduction, the start of §7 and the conclusion.

> We do not claim a more accurate pointing localiser. We show what a sub-million-parameter, image-only model can do on smart-glasses hardware, where it fails, and which design choices matter at that scale; and we compare it openly with large published models and with compact detectors trained under our protocol.

This turns C2 from a weakness into a finding. At wearable scale, a fine-tuned compact detector is a strong baseline for box accuracy. A structured model adds the hand, fingertip and direction outputs that an interface needs, at some cost in box precision.

### 3.3 Must-do before submission (documentation, not experiments)

1. **T5.** If it will not be done before 1 November, remove all of it:
   - the "Real frames" paragraph, the last row of Table 11 and Fig. 6 (§7.6);
   - "(pasted and real)" in §1 (l. 36);
   - "Section 7.6 therefore adds real frames" in §5.2 (l. 62);
   - "Real-frame behaviour is assessed in Section 7.6" in §6.2 (l. 49);
   - "Robustness to real other people's hands is tested by the real-frame check below" in §7.6 (l. 356);
   - "The real-frame check (Section 7.6) is small" in §8 (l. 28);
   - the T5 consent placeholder in §9 (l. 11).

   Replace them with one sentence in §7.6: "We did not evaluate real frames with other people's hands; this is the main open test of ownership (Section 8)."
2. **Detector recipe (§5.2, l. 71).** Fill it in from the training notebooks:
   - classes, epochs, image size, augmentation and pretrained weights;
   - the rule that picks one hand box and one target box per image;
   - the threshold rule used for Table 6.
3. **§8, l. 19, `\tbd{confirm}`.** Replace it with the composition already given in §5.1: "although 61% of EgoPoint images are composed or generated (Section 5.1), so this remains a hypothesis about real pointing."
4. **Draft macros.** Delete them from `main.tex` once nothing uses them.
5. **Excluded test images (one new sentence, arithmetic only).** Add to §5.1: "If all 43 excluded test positives were counted as failures, P@0.5 would fall by less than one point, to about 58%." This answers the "favourable exclusion" question before anyone asks it.
6. **Citations for §1.** Add sources for the market and battery statements. The sources verified in October for the other manuscript can be copied into `paper/references.bib`: IDC, Counterpoint, Yuan et al. (arXiv:2609.19793), MobileLLM and TinyissimoYOLO.

### 3.4 Title

The current title puts the least-supported claim in the most visible place. Options:

1. **Recommended if T5 is not done:** *SpatialNet: Image-Only Pointing-Target Localisation on Smart Glasses with a 0.7M-Parameter CNN.* This is close to the previous title, and every word is supported.
2. *How Small Can Pointing-Target Localisation Go? A 0.7M-Parameter CNN on Smart Glasses.* This matches the positioning in 3.2.
3. Keep "with Wearer-Aware Hand Ownership" only if T5 is filled in with a convincing result.

### 3.5 Abstract: a replacement of about 210 words

> Pointing is the most direct way to ask smart glasses about something in view ("what is that?"), but resolving which object is meant currently relies on models with hundreds of millions to billions of parameters that run off the device. We ask how much of this spatial step a model small enough for the glasses can do. SpatialNet is an image-only convolutional network (0.72M parameters, 0.23 GFLOPs, 1.6 MB in FP16) that predicts the wearer's hand, fingertip, image-plane pointing direction and the box of the pointed-at object in one forward pass. On the EgoPoint test split it localises the wearer's hand with a P@0.5 of 92.6% and the target with 58.9% (58.9–59.0% across three seeds), falling to 25.4% when the target lies far from the fingertip in the image. Published models of 226M–7B parameters trained on EgoPoint report 63.6–93.9% under their own protocols, and compact detectors that we trained at the same compute reach 59.9–67.6% with tighter boxes. Controlled experiments show what matters at this scale: input resolution is the largest lever (65.1% at 320 px with the same parameters); instance-anchored heads make the hand box and direction more precise than a plain model of the same size; training with pasted distractor hands keeps accuracy stable under synthetic hand interference; and an explicit ray prior learns mainly fingertip proximity on EgoPoint's overlap-dominated data. On a pre-production smart-glasses unit, the model runs in 60–80 ms per frame.

What changed, and why:

| Change | Fixes |
|---|---|
| One paragraph, no bullets | A1 |
| No claim that anchoring is what makes the result possible | A2 |
| Large-model numbers given "under their own protocols" | A3 |
| Untested centre-in-target claim removed | A4 |
| Pasted-hand claim tied to the training | A5 |
| "Collapses" replaced | A6 |
| Battery claim left to §7.7, where its caveats are | A7 |

### 3.6 Rewrites, section by section

Each item gives the current text, then the suggested replacement (→).

**Introduction**

- **l. 26:** "One design choice lets such a small network localise precisely: instance-anchored heads."
  → "One design choice targets precision at this size: instance-anchored heads. Against a plain model of the same size, they make the hand box and the direction more precise, although they do not change coarse accuracy (Section 7.5)."
- **l. 24:** "with presence gates for each"
  → "and a hand-presence gate. The design also has pointing and target gates, but EgoPoint provides no negative examples to train or test them."
- **l. 24:** "300 times smaller than VistaRef" → "at least 300 times smaller than VistaRef".
- **l. 17:** "to our knowledge none addresses other people's hands in view"
  → "to our knowledge none of these pointing-grounding models evaluates other people's hands in view".
- **l. 39–46, Results in brief:** delete it, or cut it to three sentences: the operating point, the comparison with detectors and the device latency. Remove the battery bullet (l. 45).
- **l. 42:** "comparable to zero-shot 7–8B multimodal LLMs ... close to a 7B model fine-tuned on EgoPoint (63.6%)"
  → "above the values reported for zero-shot 7–8B multimodal LLMs and 4.7 points below a fine-tuned 7B model, under different protocols".
- **l. 49–72, Contributions:** replace the four items with three.
  1. *A deployable, image-only pointing-target localiser for smart glasses.* A 0.72M-parameter CNN returns the hand, fingertip, direction and target box in one pass. It runs in 60–80 ms per frame on a pre-production glasses unit.
  2. *A controlled characterisation of accuracy against cost at wearable scale.* It covers:
     - matched-compute compact detectors, with a pretraining control;
     - a resolution study;
     - an end-to-end score;
     - results by fingertip–target gap;
     - a seed analysis and a duplicate audit.

     It shows that compact detectors are strong baselines for box accuracy at this scale.
  3. *Mechanism-level findings for small pointing models:*
     - resolution is the largest lever;
     - anchoring improves hand-box and direction precision, but not coarse accuracy;
     - distractor training keeps accuracy stable under synthetic interference;
     - an explicit ray prior learns mainly proximity on overlap-dominated data.
- **l. 72:** keep this sentence. It is the honest scoping, and the title should match it.

**Related Work**

- Add a short paragraph, "Compact detectors as pointing localisers". A detector fine-tuned with a single "target" class can learn which object is being pointed at, with no pointing-specific structure; this is the alternative §7.2 tests. Cite YOLO26, NanoDet-Plus and TinyissimoYOLO.
- Add one or two sentences each on HCI pointing and selection, on-device hand tracking and wearable-camera privacy (R1). This adds 8–10 references and about half a page.
- Turn the two bullet lists (l. 8–22) into two paragraphs. Drop "What SpatialNet changes" if the introduction already says it.

**Problem Definition**

- **After l. 11, add:** "EgoPoint contains only scenes 000 and 111 (Section 5.1). We therefore train and evaluate SpatialNet as a two-scene model; the pointing and target gates belong to the interface design, receive no negative examples, and are not evaluated."
- **l. 16:** "within an interactive latency" → "within an interactive latency (we take 100–200 ms per frame [cite])".
- **l. 19:** "and only when a pointing event occurs" → "and only when the wearer's hand is detected".
- **Add one sentence** defining overlap, near and far as image-plane categories.

**Results: opening**

- After the six questions, add the positioning statement from 3.2 in one sentence.

**7.1**

- **l. 84:** "We therefore claim parity at a fraction of the size, not superiority."
  → "Because the protocols differ, we read this as the same accuracy range at a fraction of the size, not as parity or superiority."
- **l. 79:** move the 57.8% re-scoring into the table note and mark it as an assumption: "If the published averages count the 425 no-target images as failures, …"
- **l. 86:** delete the comparison with VistaRef's 22.9 images/s.

**7.2**

- **l. 145:** "SpatialNet picks the right object at least as often."
  → New heading: "Target choice is similar." New body: "SpatialNet's centre-in-target rate (84.8%) is similar to or slightly above the detectors' (81.6–84.7%); we did not test these differences, so we read them as similar target choice. The detectors' advantage is box extent."

  If the per-image detector predictions are at hand, a paired bootstrap on them takes minutes and needs no retraining. Only after that should the paper say "higher".
- **l. 146:** "Only SpatialNet's accuracy does not drop with other hands in view."
  → "Under synthetic hand interference, SpatialNet's accuracy is stable; the detectors, trained without pasting, drop." Keep the rest of the bullet.
- **l. 151:** "The detectors are therefore the stronger choice when only the target box matters and a COCO-pretrained model is acceptable. SpatialNet trades …"
  → "For an application that needs only the target box, a compact detector trained on EgoPoint is the more accurate choice at this compute, which is itself useful guidance for wearable designers. SpatialNet instead returns the hand, fingertip and direction from the same pass. An interface can use these for feedback, such as drawing the ray or a fingertip cursor, which is known to help mid-air pointing [cite Mayer et al. 2018], and for hand-presence gating. Whether that is worth the loss in box precision depends on the interface, which we did not evaluate."
- **Add to the limitations:** "A detector with a keypoint head could also output the fingertip; we did not test one."

**7.3**

- **After l. 186, add:** "At threshold 0.5, the hand gate raises a false alarm on 6.6% of empty frames. An always-on interface would therefore need temporal smoothing, for example requiring several consecutive detections, which we do not evaluate."
- **l. 191–192:** replace "touched" with "overlapping (fingertip inside the target box in the image)".

**7.4**

- **l. 229:** "Beyond 224 px, far targets are limited by the direction estimate, not by the input."
  → "The plateau beyond 224 px, together with the flat direction error, suggests that the direction estimate, rather than the input, limits large-gap targets; with one run per resolution, we treat this as a hypothesis."
- **l. 231:** "Their gains are, if anything, underestimated." → "Their gains may therefore be underestimated."
- **l. 226:** delete "steady".

**7.5**

- **l. 237: the per-component metrics.**
  - If they were fixed before the ablations were run, say so explicitly: it is your strongest defence here.
  - If they were not, add: "With one seed per variant and several metrics per component, we treat these comparisons as exploratory."
- **l. 278:** "the anchored chain improves everything that requires precision"
  → "the anchored chain improves the precision of the hand box and the direction".
- **l. 285:** replace "Shorter training could widen its gaps, but it would be expected to lower P@0.5 as well; instead, the losses are confined to the precision metrics."
  → "A4 also stopped earliest (best epoch 54, against 100), so its effects include a different training length."
- **l. 304:** "Probing the trained full model explains this" → "Probing the trained full model suggests why".
- **l. 310:** "the training signal rewards a 'near the fingertip' prior"
  → "a plausible reason is the training distribution: with 70% overlap and 7% far targets, a 'near the fingertip' prior already fits most targets".
- **l. 314:** "The finding carries over to other pointing models: on touch-dominated data, giving a model the ray is not enough. The ray must be constrained to point forward, or far targets must be weighted up in training, before it can help them."
  → "We expect the same risk in other pointing models trained on overlap-dominated data. Constraining the ray to point forward, or up-weighting far targets, are hypotheses we did not test."
- **Optional:** move the backbone swap (Table 10) to the appendix and keep one sentence in §7.5.

**7.6**

- Apply item 1 of 3.3 (T5), which also covers l. 356.
- Lead the section with its two informative numbers: no-hand frames kept, and the A1 contrast. Present the 99.7% ownership accuracy as being near its ceiling, as the text already explains.

**7.7**

- **l. 421:** cite the 100–200 ms budget, and state what the 60–80 ms covers, for example "model inference only, excluding capture and pre-processing" if that is the case. Delete the VistaRef sentence.
- **l. 423–425:** replace the three battery bullets with one:
  > *Battery (one observation per mode).* Over one minute on the engineering unit, continuous inference used 9 percentage points of charge and streaming frames to the phone at 0.5 fps used 16. Each mode was measured once, the fuel gauge resolves 1%, and the unit has no production power management, so this indicates relative cost on this unit only; it is not an energy measurement or a battery-life estimate.

  If no idle baseline was recorded, say so in the same bullet. If one was, report it.
- **Add one sentence:** "We did not measure accuracy on frames from the device camera, which are portrait and are squashed to a square (Section 6.2)."

**The same claims in other sections**

- **§8, l. 8:** "makes a good always-on front end"
  → "could serve as an always-on front end once a pointing gate is trained with non-pointing hands and its output is smoothed over time".
- **§8, l. 47:** "Our measurements support this"
  → "Our one-minute battery observation is consistent with this but does not establish it".
- **§5.1, l. 10:** "so that our numbers are directly comparable with theirs"
  → "so that our test images match theirs; differences in scoring remain (Section 7.1)".
- **§6.1, l. 11:** "would, if anything, raise" → "might raise".
- **§4.4, l. 113:** "cannot switch them on by itself" → "is designed not to switch them on by itself".
- **§10, the conclusion:**
  - "which keep its boxes precise at this size" → "which make its hand box and direction more precise than a plain model of the same size";
  - "collapses into" → "is dominated by";
  - "Training with pasted, unlabelled hands teaches it to follow the wearer's hand" → "Training with pasted, unlabelled hands keeps its accuracy stable when synthetic hands are in view";
  - "costs less battery than streaming frames to a phone" → delete it, or write "in a one-minute test drained less battery than streaming frames to the phone".

### 3.7 Cut, condense, move

| Item | Action | Saves |
|---|---|---|
| Abstract | 425 → about 210 words, one paragraph | — |
| §1 "Results in brief" | Delete, or cut to 3 sentences | ~0.5 page |
| §1 bullet lists | Turn the problem bullets and the contributions into short paragraphs | Readability |
| §2 bullet lists | Two prose paragraphs | ~0.3 page |
| §7.1 bullet tree | One paragraph of findings and one paragraph of caveats | ~0.4 page |
| §7.5 backbone swap (Table 10) | Move to the appendix; one sentence in the text | ~0.5 page |
| §7.7 battery | One bullet (see 3.6) | ~0.2 page |
| Figs. 7 and 8 | Merge into one figure | ~0.5 page |
| §8 Limitations | Regroup as "Threats to validity" (see the three groups below) | Clarity |

The three groups for "Threats to validity":

- **Construct:** image-plane categories; the direction label.
- **Internal:** single seeds; different stopping epochs; bundled ablations.
- **External:** one dataset; its synthetic share; the square resize; no device frames.

Overall target: about 24–26 body pages instead of 29. Write the headline numbers out once (in the abstract and §7) and refer to them elsewhere.

### 3.8 Add without experiments: a short "Design implications" subsection in §8

This gives IMWUT readers the ubicomp payoff that C1 says is missing, using only results already in the paper.

1. **Choose the model by what the interface needs.** If only a box is needed, use a compact detector. If the interface needs the hand, fingertip and direction for feedback or gating, use a structured model (§7.2).
2. **Spend extra compute on resolution first.** For similar extra compute, a higher input resolution gains more than a larger backbone (§7.4, §7.5).
3. **Confirm large-gap targets.** Errors concentrate there, so the interface should name or highlight the target before acting on it (§7.3).
4. **Smooth over time before triggering.** Per-frame false alarms make single-frame triggering unreliable (§7.3).
5. **Keep frames on the device.** A later model needs only the target crop (§3), which limits what leaves the glasses. The battery cost on our unit is indicative only (§7.7).

### 3.9 Questions reviewers will ask, and where the paper should already answer them

| Likely question | Answer to give in the paper | Where |
|---|---|---|
| Why not just fine-tune NanoDet? | For box-only use, you should. SpatialNet adds the hand, fingertip and direction from one pass and is smaller. We report this openly. | §7.2 (last paragraph), §8 |
| Does the robustness to pasted hands come from the architecture? | No, mainly from the training recipe (A1). A detector trained with pasting is future work. | §7.2, §7.6 |
| Does it work on the glasses' own camera? | Execution and cost are measured on the device. Accuracy on device frames is not, and the camera is portrait. | §6.2, §7.7, Limitations |
| Do the 43 excluded test images flatter the result? | In the worst case, P@0.5 falls by less than one point, to about 58%. | §5.1 |
| Are the comparisons with large models fair? | They are context, not a ranking; each protocol difference is listed. | §7.1 |
| How were the detectors trained, and how is one box selected? | The full recipe and the box-selection rule. | §5.2 |
| What do the pointing and target gates do? | Nothing measurable on EgoPoint; they are design for future data. | §3, §4.4 |
| Is the battery result real? | One observation per mode on a pre-production unit; relative cost only. | §7.7 |
| Ablations with one seed? | Paired bootstrap plus the seed-spread rule. Effects close to the rule are labelled supporting, and P@0.5 differences are not interpreted. | §7.5 |
| Why keep a ray branch that does not help? | It is part of the deployed, measured model. It costs no coarse accuracy, and the analysis of what it learned is itself a finding. | §7.5 |
| Do test images duplicate training images? | Audited. Conclusions hold except two P@0.7 effects, which are flagged in Table 9. | §5.2, §7.5 |
| Why 160 px when 320 px is better? | The device budget. 160 px is the model measured on the glasses; 224 px and 320 px quantify the headroom. | §7.4 |

### 3.10 Keep exactly as is

- The matched-compute detector comparison, including NanoDet's lead.
- The pretraining control (COCO, ImageNet, scratch).
- The duplicate audit, and the § footnotes in Table 9.
- Every cross-protocol caveat in Table 4.
- The seed analysis and the "Retraining" row.
- The A3 negative result: it is one of the paper's most useful findings.
- The "deliberately selected" wording in the figure captions.
- The statement in §4.4 that the pointing and target gates are not evaluated.

### 3.11 Expected effect on the score

| Version | Expected score |
|---|---|
| As it stands | 4 / 10 |
| With sections 3.3–3.8 applied (writing only) | 5–6 / 10: a clean major revision. Reviewers can argue about the contribution, but not about overclaims. |
| Revision round with T5 and some accuracy measured on device frames | 6–7 / 10 |
| Also with a small user task on the glasses | 7–8 / 10 |
