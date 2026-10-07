# Submission manuscript

A complete 10-page, venue-neutral two-column revision, including references. The original paper and experiment outputs are preserved.

- main.pdf: compiled manuscript.
- main.tex and sections/: editable LaTeX source.
- figures/: publication figures.
- references.bib: source bibliography (only cited entries are printed).
- evidence.json: selected numerical summaries and stored paired intervals, traced to the existing outputs.
- example_provenance.json: source image IDs, annotations, saved predicted boxes and checked IoUs for all eight displayed examples.
- METHOD_NOTES.md: detailed implementation definitions omitted from the compact manuscript.
- EDITORIAL_NOTES.md: claim choices, comparison qualifications and provenance.
- scripts/: audit, figure and document-building utilities.
- qa/: local build logs and rendered pages; not part of the submission.

T5 is the only red manuscript placeholder. It covers the real multi-hand evaluation and its data/consent description. No user study is required or proposed as a missing submission item.

## Build the manuscript

From this directory, run:

    powershell -ExecutionPolicy Bypass -File build.ps1

This compiles LaTeX and bibliography only. It does not train a model, run inference, or recompute experiments. MiKTeX/TeX Live with the listed LaTeX packages and BibTeX is required. The PDF is anonymous and has no venue branding.

## Figure provenance

The opening figure contains four deliberately selected successful predictions. The failure figure contains four deliberately selected complex scenes. All boxes, fingertip positions and displayed IoUs are taken from the saved main-run test CSV; annotations independently verify the displayed IoUs within CSV precision. Original dataset images are shown in normalized square coordinates. Direction is omitted from these visual examples to keep the overlays clear.

Images remain subject to the source dataset's terms. No raw dataset is copied into this folder.

## Scope of this revision

All training and aggregate evaluation outputs are the existing repository results. An optional single-image CPU inference check was attempted during figure preparation; the final figures consistently use the saved evaluation outputs rather than mixing predictions from different execution conditions. No training was performed.
