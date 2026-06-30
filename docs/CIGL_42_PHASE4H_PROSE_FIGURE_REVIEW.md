# CIGL_42 — Phase 4H Prose / Figure / Table Polish + Review-PDF Review

> Phase 4H summary (writing/packaging only; no GPU, no training, no experiments, no ablation, no new
> dataset, no λ-grid, no edge-CMI, no SOTA, no CITA/DualPC/Tri-CMI changes, **no PDF/aux/bbl/log committed**,
> no invented citations, **no fabricated figure data**).

## Prose summary

Section prose keeps the audited bounded wording (posterior-KL proxy / not unbiased CMI; partial reduction
not elimination; source-task retention criteria; target labels evaluation-only; graph/node only; edge-CMI
out of scope; dynamic-edge unsupported; no SOTA/beyond-MI). The v0.6 changes are structural (tables and
figures wired in, cross-references) rather than claim-level; a claim-fidelity check confirmed no drift.

## Table summary

The five tables (T1 method/protocol, T2 audit, T3 BNCI2014_001, T4 BNCI2015_001, T5 negative results) are
now `\input` into the document in order (previously they existed but were never included, so the v0.5 PDF
had no tables). Textual "Table~N" mentions are now `\ref{tab:...}`. Edge status is explicit ("skipped / no
edge-CMI"). No numbers were changed or invented.

## Figure asset status

- **F2** (reduction vs source-task retention) and **F3** (audit vs permutation null): generated **from real
  values** (per-fold confirmation JSON; 3A-H audit values), committed as PNG (+SVG) and **embedded** in the
  review PDF. F2 annotates the fold9 retention miss; F3 matches Table~2 exactly.
- **F1** (pipeline) and **F4** (decision flow): schematic SVG drafts, shown as labeled placeholder floats (no
  SVG→PDF converter installed; final embed is a human/TikZ step). See `paper/cigl_latex/FIGURE_STATUS.md`.
- No figure implies unobserved data.

## Compile smoke status: **SUCCESS**

`pdflatex → bibtex → pdflatex ×2` (TeX Live 2024). All passes exit 0; 11 bib entries, no errors; **no
undefined refs/cites; no missing figure files**; `_build/main.pdf` ~547 KB, **12 pages** (tables + figures
now included). Generated output only under gitignored `_build/`. See `REVIEW_PDF_SMOKE_SUMMARY.md`.

## Remaining blockers

1. Final F1/F4 embed (SVG→PDF or TikZ) at venue-template time.
2. Human prose pass; venue template/style.
3. Minor Brunner Graz-2a data-description URL.
4. Authorized full submission PDF build.

## Confirmations

- No GPU / training / experiments / ablation / new dataset / λ-grid; no edge-CMI; no SOTA; no
  CITA/DualPC/Tri-CMI changes; **no generated PDF/aux/bbl/log committed**; no generated result tables
  committed; no fabricated figure data (F2/F3 from existing JSON/values only).
- Validation: py_compile OK; `test_collect_cigl_evidence_tables`, `test_cigl_manuscript_claims`, updated
  `test_cigl_latex_package` pass; collect dry-run OK; `git ls-files` shows no tracked LaTeX artifacts.

## Recommendation (pending reviewer)

**A — ready for human reading / venue-template decision.** The draft now reads as a paper (rendered tables,
data-backed figures, working bibliography). No experiment is required; `graph_010` vs `node_010` stays
optional and unauthorized.
