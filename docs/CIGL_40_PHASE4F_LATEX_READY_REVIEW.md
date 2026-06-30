# CIGL_40 — Phase 4F Citation Closure + LaTeX-Ready Package Review

> Phase 4F summary (writing/packaging only; no GPU, no training, no experiments, no ablation, no new
> dataset, no λ-grid, no edge-CMI, no SOTA, no CITA/DualPC/Tri-CMI changes, **no PDF compiled**, no invented
> citations). Artifacts under `paper/cigl/` and the new `paper/cigl_latex/`.

## Citation closure status

**4/10 resolved** (reviewer-verified, now with full DOIs in `REFERENCES_DRAFT.bib`):
- BNCI2014_001 → `tangermann2012review` (*Front. Neurosci.* 6:55, DOI 10.3389/fnins.2012.00055) +
  `brunner2008graz2a` (Graz-2a data description).
- BNCI2015_001 → `faller2012autocalibration` (*IEEE TNSRE* 20(3):313–319, DOI 10.1109/TNSRE.2012.2189584).
- MOABB → `jayaram2018moabb` (*JNE* 15(6):066011, DOI 10.1088/1741-2552/aadea0).
- EEGNet → `lawhern2018eegnet` (*JNE* 15(5):056013, DOI 10.1088/1741-2552/aace8c).

The **dataset-primary citation blocker is cleared.**

## Remaining citation TODOs (6/10, not fabricated)

Schirrmeister (vol/pages/DOI), DGCNN (exact venue/year/DOI), Li 2018 (author list/DOI), CCMI (vol/pages) —
required before submission; RGNN (arXiv:1907.07835) and LGGNet (arXiv:2105.02786) are arXiv-citable now and
may be upgraded. All are visible as `\todoverify` in the `.tex` and `% TODO` in the `.bib`. See
`CITATION_TODO_QUEUE.md`.

## BibTeX draft status

`paper/cigl/REFERENCES_DRAFT.bib`: 11 entries, verified-identity only; unverified fields are inline `% TODO`
comments (never fabricated). Header warns "Draft references; unresolved TODOs remain before submission."

## LaTeX package status

`paper/cigl_latex/` created and self-consistent (not compiled, no PDF, neutral `article` class):
- `main.tex` (abstract + `\input` of 7 sections + `\bibliography{../cigl/REFERENCES_DRAFT}`).
- `sections/01..07*.tex` — faithful LaTeX conversion of the v0.3 prose; **claims unchanged** (claim-fidelity
  audit found zero drift).
- `tables/table1..5*.tex` — LaTeX `tabular` drafts (existing values + fold-level bootstrap CIs).
- `figures/FIGURE_ASSET_PLAN.md` — F1–F4 plans; no images generated.
- `README.md` — hard rules + build-only-when-authorized note.

## Confirmations

- No GPU / no training / no experiments / no ablation / no new dataset / no λ-grid; no edge-CMI; no SOTA;
  no CITA/DualPC/Tri-CMI changes; **no PDF compiled**; no generated result tables committed (gitignored).
- Validation: py_compile OK; `test_collect_cigl_evidence_tables`, `test_cigl_manuscript_claims`, and the new
  `test_cigl_latex_package` pass; collect dry-run OK; package-presence checks pass.

## Are new experiments recommended?

**No.** The paper stands without any new experiment (confirmed by the Phase 4E adversarial area-chair judge
and unchanged here). The `graph_010` vs `node_010` mechanism ablation remains optional and unauthorized.

## Recommendation (pending reviewer)

**A — ready for human prose / LaTeX editing.** Open items: 6 remaining citation fields, F1–F4 asset
generation (F2/F3 from existing JSON), a venue template, and an authorized PDF compile.
