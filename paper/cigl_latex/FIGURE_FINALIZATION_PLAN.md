# CIGL Figure Finalization Plan (Phase 4I)

> Plan for taking the v0.6 draft figures to camera-ready. Current status is in `FIGURE_STATUS.md`. No new
> data; F2/F3 use existing JSON/audit values only.

## F1 — Pipeline schematic
- **Current:** hand-drawn SVG (`figures/fig1_pipeline_draft.svg`), shown as a placeholder float (no
  SVG→PDF converter installed here).
- **Final form:** redraw in **TikZ** (vector, template-native, no external converter) — stem → static
  adjacency / ChebConv → node\_z, graph\_z → logits; dashed Step-A probes; Step-B `λ_g R_g + λ_n R_n`;
  explicit `edge_logits=None` / no-edge annotation.
- **Source data:** none (architecture).
- **Status:** manual; **must check** that it claims no accuracy/edge result and matches §3 exactly.

## F2 — Leakage reduction vs source-task retention (data-backed)
- **Current:** generated PNG+SVG from real per-fold values; embedded.
- **Final form:** regenerate at 300+ dpi PDF/PNF from the **per-fold confirmation JSON**
  (`results/cigl/phase3a_dgcnn_gn_multifold_confirmation/BNCI2014_001_*summary.json` folds 1–8;
  `.../phase3a_dgcnn_gn_second_dataset_confirmation/BNCI2015_001_*summary.json` 12 folds). x = graph/node KL
  reduction %, y = source bAcc drop; gate line 0.02; fold9 annotated.
- **Status:** **auto-generated** (matplotlib, values only). **Must check** axis labels say *retention*
  (drop), legend distinguishes datasets + graph/node, fold9 annotation present.

## F3 — Audit vs permutation null (data-backed)
- **Current:** generated PNG+SVG; embedded.
- **Final form:** regenerate bars (graph 1.261 vs 0.16; node 0.521 vs 0.034) + (optional) per-electrode
  node-leakage map inset if the saved map array is recovered.
- **Source data:** 3A-H audit values (Table 2 / CIGL_25); node-map array if available.
- **Status:** bars **auto-generated**; node-map inset **needs the saved array** (else omit). **Must check**
  values equal Table 2 and the caption says "proxy, not unbiased CMI".

## F4 — Negative-results decision flow
- **Current:** hand-drawn SVG (`figures/fig4_decision_flow_draft.svg`), placeholder float; the section
  caption no longer uses internal phase codes.
- **Final form:** redraw in **TikZ** as a clean top-to-bottom flow with pass/fail badges; the dynamic-edge
  branch labeled "consistent with fingerprint risk, not causal"; **remove internal codes** (3A-*, CIGL\_NN)
  for the camera-ready — use descriptive step names only.
- **Source data:** none (qualitative gates; numbers live in Tables 2–5).
- **Status:** manual; **must check** no quantitative claim beyond documented values (src ≈ 0.33).

## Cross-cutting checks before submission

1. Replace placeholder floats (F1, F4) with embedded TikZ/PDF once a venue template is chosen (no external
   converter dependency).
2. Re-derive F2/F3 from JSON in a single committed script for reproducibility (values only; no new runs).
3. Ensure every figure caption carries the bounded wording (proxy not CMI; retention not accuracy;
   edge-CMI out of scope) and **no figure implies unobserved data**.
4. Remove internal project codes (3A-*, CIGL\_NN, raw flag names) from all final figures.
5. Confirm colour-blind-safe palette and grayscale legibility.
