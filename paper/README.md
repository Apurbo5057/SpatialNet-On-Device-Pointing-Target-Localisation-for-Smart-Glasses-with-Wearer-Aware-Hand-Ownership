# SpatialNet paper (IMWUT draft)

ACM `acmart` class, `acmsmall` format, `\acmJournal{IMWUT}`, anonymous review mode. The current draft is 22 pages, including references and appendix.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`latexmk` needs Perl, which is not installed here, so run the four commands above. `acmart.cls` was generated from the official acmart source in `template/acmart-primary/`.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Preamble, title, abstract, teaser, draft macros, section includes |
| `sections/01-introduction.tex` … `10-ethics-conclusion.tex`, `appendix.tex` | One file per section |
| `references.bib` | All references. The 2026 arXiv entries were checked in October 2026; switch them to venue versions if they get published. |
| `figures/architecture.pdf` | Fig. 2, exported from the Canva design (link below), placed as a landscape page |
| `figures/teaser.pdf`, `figures/ownership_examples.pdf` | Figs. 1 and 3, made by `make_teaser.py` and `make_ownership_examples.py` |
| `figures/thumbs/`, `make_thumbnails.py`, `make_ray_inset.py` | Images placed inside the Canva diagram |
| `scripts/simple_references.py` | The "Simple references" rows of Table 4, computed from `predictions/test.csv` |

Architecture diagram in Canva: <https://canva.link/senozgc7x6ep7k9> (design `DAHXR61pElc`). After editing it, export it as PDF to `figures/architecture.pdf`.

## Red placeholders

| Macro | Meaning |
|---|---|
| `\tbd{...}` | Text to write or confirm |
| `\xx`, `\xxpm` | A number, or a mean ± std, to fill in |
| `\phfig{w}{h}{...}` | A figure to insert |

Find them all with `grep -n "tbd\|\\\\xx\|phfig" main.tex sections/*.tex`. Delete the macro definitions in `main.tex` before submitting, so that a forgotten placeholder breaks the build instead of being printed.

## Where each result goes

| Source | Goes to |
|---|---|
| Main run (`runs/main_run/`, `best_model.pth`, seed 42) | **Filled in:** Tables 4–7 (SpatialNet rows), Figs. 1, 4 and 6, §6.1, §7.2, §7.6, abstract, introduction and conclusion. Figs. 1 and 6 are re-made from the checkpoint by `make_teaser.py` and `make_qualitative.py`. If the main run is redone, re-run those scripts and update these numbers. |
| T2 on-glasses measurements | §6.2 pipeline and measurement; Table 8; abstract latency |
| T3 ablations A1–A3 (`docs/paper_plan.md` runbook) | Table 6 with paired-bootstrap intervals; §7.3 text; then revise the claims in §9 |
| T4 seeds 1 and 2 | **Filled in:** Table 4 "mean ± std, 3 seeds" row; Table 7 "Retraining" row; Fig. 5 bands; §7 intro |
| Benchmarks (`docs/BENCHMARK_RESULTS (2).md`, 2026-10-11) | **Filled in (v1 only; the benchmark's "v2" is deliberately not reported):** §5 end-to-end score, detector protocol and duplicate check; §7.2 and Tables 5–6 (YOLO26n, NanoDet-Plus-m); A3/A4 duplicate caveats and full-test ray probe in §7.4; backbone swap (Table 10); §7.5 detectors under pasted hands; abstract, introduction, discussion, conclusion. One red `\tbd` remains in §5: the detector training recipe and box-selection rule. Paired intervals against detectors exist only for v2, so v1-vs-detector differences are reported as point estimates. |
| T5 real multi-person frames | Table 7 last row; Fig. 5; §7.4 |

## Before submission

- **Status (2026-10-08):** the user study (T6) was removed, and every red item except T5 is resolved.
- **Only T5 remains red:** the real-frame paragraph and Table 8 last row in §7.5, Fig. 6, and the consent sentence in §10.
- **After T5:** delete the draft macros in `main.tex`, so that a forgotten placeholder breaks the build.
- **Device:** §6.2 gives generic hardware only (no chip or vendor names), for confidentiality and anonymity.
- **Dataset:** EgoPoint access is limited to certified researchers. Keep the converted Kaggle datasets private; §10 says we don't redistribute.
- **Camera-ready:** add the code and weights link (§10 promises release upon publication).
