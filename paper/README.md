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
| T4 seeds 1 and 2 | Table 4 "± std over 3 seeds" row |
| T5 real multi-person frames | Table 7 last row; Fig. 5; §7.4 |
| T6 user study | §8. If T6 is not run, delete §8 and add deployment observations to §9 |

## Before submission

- **Device details.** `docs/Glass Spec.png` looks like an internal spec sheet. Confirm what may be published about the developer unit (§6.2, §10), and anonymise it for review.
- **Dataset access.** The EgoPoint-Ground paper says access is limited to certified researchers under a licensing system. Keep the converted Kaggle dataset private, and state the licence in §5 and §10.
- **Split.** The EgoPoint-Ground paper states a 7:2:1 train:val:test ratio, but the released folders are about 7:1:2 and match VistaRef's 3,072-image test split. Confirm with the authors and update §5.
- **Negatives in published numbers.** VistaRef and PointVG-R average over all 3,072 test images, including 425 without a target, and don't say how those are scored. Ask the authors if possible; until then, the paper claims parity with zero-shot LLMs, not superiority.
- **CCS concepts.** Regenerate the CCS XML at <https://dl.acm.org/ccs>.
- **Numbers.** Every number must come from `metrics.json`, the T2–T6 logs, or the cited papers. The VLM comparison uses published zero-shot results (PointVG-R, EgoPoint-Ground), so there is no own VLM run (former T1).
- **Anonymity.** Anonymise code and data links.
