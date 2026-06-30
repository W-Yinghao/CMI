# CIGL Manuscript v0.6 — Review Notes (Phase 4H)

> Self-review of the v0.5 → v0.6 transition (prose/table/figure polish + review-PDF smoke; writing/packaging
> only). Externally-facing summary: `docs/CIGL_42_PHASE4H_PROSE_FIGURE_REVIEW.md`.

## What changed from v0.5

- **Tables now render.** The five table `.tex` files were not `\input` before (the v0.5 PDF had no tables);
  they are now `\input` in order (T1 in §4, T2–T4 in §5, T5 in §6) and all textual "Table~N" mentions became
  `\ref{tab:...}` cross-references. No numbers changed.
- **Figures wired in.** Four figure floats added with captions/labels: F1 pipeline + F4 decision-flow as
  labeled placeholder floats (schematic SVG drafts), F2 reduction-vs-retention and F3 audit-vs-null
  **embedded as PNGs generated from real data** (per-fold confirmation JSON / 3A-H audit values).
- **Review PDF compiles** to 12 pages (was 8) with all tables + figures, no undefined refs/cites, no missing
  files.
- New: `figures/{fig1_pipeline_draft.svg, fig2_*_draft.{png,svg}, fig3_*_draft.{png,svg},
  fig4_decision_flow_draft.svg}`, `FIGURE_STATUS.md`, `REVIEW_PDF_SMOKE_SUMMARY.md`; `FIGURE_ASSET_PLAN.md`
  updated.

## Prose / claims

The bounded claim is unchanged: posterior-KL proxy (not unbiased CMI), partial graph/node leakage reduction,
source-task retention criteria, target labels evaluation-only, graph/node only, edge-CMI out of scope,
dynamic-edge unsupported, no SOTA, no beyond-MI. The v0.6 edits are structural (tables/figures/refs); the
section prose carries the same audited wording, so no claim drift is introduced.

## Table changes

T1–T5 unchanged in content; now rendered as floats with `\ref` cross-references. Edge status remains
explicit ("skipped / no edge-CMI") in T1–T5.

## Figure asset status

- F2/F3: **draft from real data** (PNG embedded + SVG), values traceable to CIGL_25/29/31.
- F1/F4: **schematic SVG drafts**, placeholder floats in the PDF (no SVG→PDF converter installed here; final
  embed is a human/TikZ step). See `FIGURE_STATUS.md`.

## Compile smoke result

**SUCCESS** — `pdflatex → bibtex → pdflatex ×2`, all exit 0, 11 bib entries, no undefined refs/cites,
`_build/main.pdf` ~547 KB, **12 pages**. Generated output stays in gitignored `_build/`; no PDF/aux committed.

## Remaining blockers before submission (all writing/packaging — no GPU)

1. Final F1/F4 embed (SVG→PDF or TikZ) once a venue template is chosen.
2. Human prose pass for flow at submission length.
3. Venue template / style; minor Brunner Graz-2a data-description URL.
4. Authorized full submission PDF build.

## Is any experiment paper-critical?

**No.** Bounded claim fully supported by CIGL_25/29/31; `graph_010` vs `node_010` ablation remains optional.

## Recommendation

**A — ready for human reading / venue-template decision.** The draft is now readable with rendered tables
and data-backed figures; no experiment is required.
