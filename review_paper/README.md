# SpatialNet paper, compact review version

This is the version to share with collaborators for review. It is a shortened rewrite of `revised_paper/`, which stays unchanged as the full record. No experiment, number or claim was added: every number here appears in `revised_paper/` and comes from the same runs.

## What changed compared with `revised_paper/`

| | `revised_paper/` | `review_paper/` |
|---|---|---|
| Total pages | 32 | 29 |
| Main text | pages 2–27 | **pages 2–13** |
| Tables in the main text | 13 | 6 |
| Numbers in the abstract | 14 | 5 |

- **Abstract, rewritten.** Like the VistaRef, PointVG-R and EgoPoint-Ground abstracts, it follows motivation → gap → approach → key results → lesson, but with our own motivation:
  - glasses cannot afford large models;
  - streaming frames costs battery, latency, server cost and bystanders' privacy;
  - so we ask what a model small enough for the glasses can do.

  It gives five numbers (size, the 300–10,000× ratio, latency, 59% against 64–94%) and does not mention far objects.
- **Introduction:** shorter, motivation first, with three contributions.
- **Method:** in prose; all equations, the parameter table, the loss table and the augmentation table moved to Appendix A.
- **Results:** four core sections.
  - one combined comparison table with large models, compact detectors and SpatialNet (Table 2);
  - target choice and the recognition test (Table 3);
  - what matters at this scale, with a compact ablation table (Table 4);
  - on the glasses, the real-frame check, and failures.
- **Moved to Appendix C:** the full published-model table, the benchmark detector table, the controlled runs, the COCO test, ownership under pasted hands, accuracy by fingertip–target gap, recognition details, the resolution details, the full ablations and their figure, the ray-prior probe with the remedies (E1–E3), the backbone swap, the IoU curves and the qualitative examples.
- **Far objects** appear in the main text only as two honest caveats (the recognition test and the design implication on confirmation); their analysis is in the appendix.

## Still to fill in (red text)

The same three items as in `revised_paper/`, all depending on data from you or Samsung:

1. **Device measurements** (§5.4, Table 5): still written as single one-minute observations. Replace them once sections A–D of `docs/Samsung_Device_Measurement_Request.md` come back.
2. **Real-frame check** (§5.5, Table 6, Fig. 4, the §7 consent text, and the `%%` comment in the abstract): from section G of the request.
3. **Training recipe of your benchmark detectors** (Appendix C.1): it covers the NanoDet-Plus-m row of Table 2.

`revised_paper/README.md` explains how to score the real frames and what to do if section G cannot be done in time. The same steps apply here; the section numbers differ.

## Notes

- The teaser image (Fig. 1) still labels panel (c) "Large gap". The label is part of the image file, which was copied from the other manuscript; the caption no longer uses the term.
- References are shared with `revised_paper/` and were verified there (see its README).

## Build

```bash
cd review_paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
