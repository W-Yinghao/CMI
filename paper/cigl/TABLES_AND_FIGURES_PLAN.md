# CIGL Tables & Figures Plan (Phase 4E / v0.3)

> No figures generated yet. Each item: source, content, claim supported, claim it must NOT support,
> status (ready / needs-generated-table / needs-plotting / needs-reviewer-decision).

## Tables

### T1 — Method / config / protocol
- **Source:** CIGL_32, CIGL_36.
- **Columns:** backbone (DGCNN static adj), loss `CE+λ_g R_g+λ_n R_n`, λ_g=λ_n=0.010 / λ_edge=0, paradigm
  (MotorImagery), datasets (BNCI2014_001 4-class chance 0.25; BNCI2015_001 binary chance 0.50), resample
  128, window 0.5–3.5 s, seeds 0–2, n_perm 50, setting source-only.
- **Supports:** the fixed, source-only protocol. **Not:** any accuracy/SOTA comparison. **Status:** ready.

### T2 — DGCNN leakage audit (3A-H, BNCI2014_001 fold-0)
- **Source:** CIGL_25 + `..._dgcnn_leakage_audit_summary.json`.
- **Columns:** object (graph/node), kl_mean, permutation_mean, permutation_p, clears_null (per seed),
  node-map stability (mean_corr, null_q95, above_random).
- **Supports:** leakage exists + is spatially stable. **Not:** edge leakage (skipped); not control.
  **Status:** ready (collect script).

### T3 — BNCI2014_001 confirmation (3A-J, primary folds 1–8)
- **Source:** CIGL_29 + `..._multifold_summary.json` (via `collect_cigl_evidence_tables.py`).
- **Columns:** fold, ERM src, reg src, src drop, graph KL e→r (red%), node KL e→r (red%), clears, retain,
  pass; footer = 8/8 criteria, fold-0 = dev (excluded).
- **Supports:** within-dataset confirmation at task retention. **Not:** cross-dataset/SOTA. **Status:**
  needs-generated-table (CSV/MD exists locally; gitignored).

### T4 — BNCI2015_001 confirmation (3A-K, 12 folds)
- **Source:** CIGL_31 + `..._2nd_dataset_summary.json`.
- **Columns:** as T3 + target drop (eval-only); footer = three-layer verdict
  (`confirmed_with_target_guardrail=true`), source retained 11/12.
- **Supports:** second-dataset confirmation + target guardrail. **Not:** elimination/SOTA. **Status:**
  needs-generated-table.

### T5 — Negative-results summary
- **Source:** CIGL_18 (3A-R), CIGL_21 (3A-S), CIGL_23 (3A-G).
- **Columns:** phase, finding (GraphCMINet near-chance; protocol learnable; dynamic-edge overfit), what it
  rules out, why it shapes the method.
- **Supports:** scope justification. **Not:** a positive method claim. **Status:** ready.

## Figures (none generated yet)

### F1 — CIGL pipeline schematic
- **Source:** CIGL_32 method. **Visual:** raw EEG → DGCNN temporal stem → static adjacency → ChebConv →
  node_z → graph_z → logits; Step-A posteriors `q_g`/`q_n`; penalties `R_g`/`R_n`.
- **Supports:** method clarity. **Not:** edge term (must show edge_logits=None). **Status:** needs-plotting.

### F2 — Leakage reduction vs task retention scatter
- **Source:** T3/T4 per-fold. **Visual:** x = graph (or node) KL reduction %, y = source-acc drop; both
  datasets; guardrail lines at drop 0.02 / reduction 30%.
- **Supports:** "reduction at retention". **Not:** elimination (points are not at 100% reduction).
  **Status:** needs-plotting.

### F3 — Graph/node leakage audit schematic
- **Source:** CIGL_25 audit. **Visual:** frozen features → conditional-domain probe vs within-label
  retrained permutation null → kl/p; node-leakage map over electrodes.
- **Supports:** audit validity. **Not:** unbiased-CMI framing. **Status:** needs-plotting (+ node-map
  needs-reviewer-decision: optional electrode map).

### F4 — Negative-result decision flow
- **Source:** CIGL_33/§6. **Visual:** GraphCMINet fail → decoder sanity → backbone redesign → only static
  DGCNN passes → graph/node audit → regularizer.
- **Supports:** methodology/scope. **Not:** a results claim. **Status:** needs-plotting.

## Notes
- Generated CSV/MD tables (`results/cigl/paper_tables/`, via `scripts/collect_cigl_evidence_tables.py`) are
  **gitignored** unless explicitly approved.
- No figure asset is produced in the Phase 4x writing stages (writing only).
