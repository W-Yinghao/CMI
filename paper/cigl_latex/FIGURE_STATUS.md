# CIGL Figure Status (Phase 4H / v0.6)

> Status of each figure asset. **No figure implies unobserved data.** F2/F3 are generated from existing
> values only; F1/F4 are schematic. Captions live in the section `.tex`; data-honesty notes below.

| fig | type | asset(s) committed | embedded in review PDF? | data source | status |
|---|---|---|---|---|---|
| **F1** pipeline | schematic | `figures/fig1_pipeline_draft.svg` | placeholder float (text) | none (architecture) | DRAFT — needs SVG→PDF or TikZ redraw for final embed |
| **F2** reduction vs retention | data plot | `figures/fig2_reduction_vs_retention_draft.png` (+ `.svg`) | **yes** (`\includegraphics`) | per-fold confirmation JSON (CIGL_29/31) | DRAFT from real data |
| **F3** audit vs null | data plot | `figures/fig3_audit_null_draft.png` (+ `.svg`) | **yes** (`\includegraphics`) | 3A-H audit values (CIGL_25 / Table~2) | DRAFT from real data |
| **F4** decision flow | schematic | `figures/fig4_decision_flow_draft.svg` | placeholder float (text) | none (qualitative gates) | DRAFT — needs SVG→PDF or TikZ redraw for final embed |

## Data-honesty notes

- **F2** is one point per LOSO fold: x = graph/node KL reduction (%) of `graph_node_010` vs fold/seed-matched
  ERM; y = source bAcc drop vs ERM. BNCI2014_001 uses primary folds 1–8 (fold-0 dev excluded); BNCI2015_001
  all 12 folds; fold9 (+0.024, the one retention miss) is annotated. No smoothing, no synthetic points.
- **F3** shows the observed posterior-KL proxy vs the within-label permutation null for graph (1.26 vs 0.16)
  and node (0.52 vs 0.034) on BNCI2014_001 fold-0 — the exact values reported in Table~2 / CIGL_25.
- **F1/F4** are schematic only (no numbers); F4's quantitative claims live in Tables 2–5.

## Why F1/F4 are not embedded as images

No SVG→PDF/PNG converter (`rsvg-convert`/`inkscape`/`cairosvg`) is installed in this environment, and
`*.pdf` is git-ignored, so the hand-drawn SVGs cannot be rasterized to a committable raster here. They are
rendered as **labeled placeholder floats** in the review PDF and shipped as committed `.svg` draft assets;
final embedding (SVG→PDF, or a TikZ redraw) is a human/venue-template step.

## Reproduce F2/F3

`python` (matplotlib) over the per-fold JSON under
`results/cigl/phase3a_dgcnn_gn_{multifold,second_dataset}_confirmation/` — values only, no new runs.
