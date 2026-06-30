# CIGL Figure Asset Plan (Phase 4F / v0.4)

> Plans only — **no figure images generated** in this phase. Captions are in
> `../../cigl/FIGURE_CAPTIONS_DRAFT.md`. Each plan states the data source, visual elements, what it supports,
> what it must NOT be used to claim, and whether it is auto-generatable from existing JSON or hand-drawn.

## F1 — CIGL pipeline schematic
- **Data source:** none (architecture diagram); `cmi/models/graph_task_backbones.py`, `cmi/train/trainer.py`.
- **Visual:** stem → static adjacency → ChebConv → `node_z` → `graph_z` → logits; dashed Step-A probe boxes;
  `λ_g R_g`, `λ_n R_n` penalty arrows; explicit "edge_logits = None / no edge term" annotation.
- **Supports:** the graph/node-only method on a static-adjacency backbone; where the proxies attach.
- **Must NOT claim:** any edge-CMI / dynamic-edge method; any accuracy/SOTA result.
- **Generation:** **hand-drawn** (TikZ or vector tool). Not auto-generatable.

## F2 — Leakage reduction vs source-task retention gate
- **Data source:** per-fold JSON (`results/cigl/phase3a_dgcnn_gn_multifold_confirmation/*.json`,
  `.../phase3a_dgcnn_gn_second_dataset_confirmation/*.json`); summarized in `STATISTICAL_SUMMARY_DRAFT.md`.
- **Visual:** scatter, one point per LOSO fold; x = graph (or node) KL reduction %, y = source bAcc drop;
  two marker styles (2a folds 1–8 / 2015 12 folds); reference lines at retention gate (0.02) and ≥30%
  reduction; fold9 (+0.024) annotated.
- **Supports:** "partial reduction while meeting the retention gate"; the honest fold9 exception. The y-axis
  is a **retention** axis (near-zero drop = task retained), **not** an accuracy gain.
- **Must NOT claim:** leakage elimination (no point at 100%); accuracy improvement; SOTA.
- **Generation:** **auto-generatable** from existing per-fold JSON (matplotlib; no new runs). Mark draft.

## F3 — Graph/node leakage audit with retrained permutation null
- **Data source:** `docs/CIGL_25` (3A-H); `cmi/eval/graph_leakage.py::audit_graph_node_objects`.
- **Visual:** grouped bars (observed vs null KL for graph and node) with significance; per-electrode
  node-leakage map inset (corr ≈ 0.945 across seeds).
- **Supports:** the audited leakage is significant and spatially stable; the null is a proper control.
- **Must NOT claim:** an unbiased CMI value (proxy); edge leakage (skipped).
- **Generation:** bars **auto-generatable** from the audit JSON; node-map inset needs the saved map array.

## F4 — Negative-results decision flow
- **Data source:** `docs/CIGL_18/21/23/25`, `docs/CIGL_33`.
- **Visual:** top-to-bottom decision flow with pass/fail badges (3A-R → 3A-S → 3A-G → 3A-H → 3A-I/J/K); the
  "dynamic-edge overfit" branch marked "consistent with fingerprint risk, not causal"; surviving path
  highlighted as the deliberate scope.
- **Supports:** the methodology and scope; negatives as positive, method-shaping evidence.
- **Must NOT claim:** that `A(x)` is causally the leakage source; any edge-CMI/dynamic-edge method.
- **Generation:** **hand-drawn** (TikZ/flowchart). Not auto-generatable.

## Notes
- No image assets are committed in Phase 4F. F2/F3 (bars) can be produced later from existing JSON with no
  new experiments; F1/F4 are schematic and hand-drawn. All figures inherit the bounded claims above.
