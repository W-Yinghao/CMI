# CIGL Acceptance Criteria

This file defines pass/fail gates. A phase does not pass because code exists; it passes because the required evidence exists.

---

## Gate 1 — Backbone validity

### Required evidence

- `GraphCMINet.forward(x)` returns `(logits, graph_Z)`.
- `GraphCMINet.forward_graph(x)` returns `(logits, graph_Z, node_Z, edge_logits)`.
- Shape checks pass for channel counts 19, 22, 32, 62.
- All outputs are finite.
- One backward pass works.
- GraphCMI-ERM trains for at least one mini epoch on synthetic tensors.
- Adjacency is non-degenerate:
  - not all zero;
  - not all identity;
  - not constant across samples;
  - no NaNs or exploding degree normalization.

### Pass condition

All tests pass on CPU. No full EEG run is required.

### Fail action

Fix backbone and tests before touching node/edge CMI experiments.

---

## Gate 2 — Leakage hypothesis

### Required evidence

On at least one real source-only dataset/protocol:

- GraphCMI-ERM produces measurable graph leakage \(I(Z_g;D\mid Y)\).
- `node_Z` produces measurable node leakage.
- `edge_logits` produces measurable edge leakage \(I(A;D\mid Y)\).
- Edge leakage is above within-label domain permutation null.
- Node leakage map has seed stability above a trivial random-map baseline.

### Pass condition

At least two of the three leakage objects — graph, node, edge — show nontrivial conditional domain leakage, and edge leakage is not zero/noise.

### Fail action

If edge leakage is absent, do not claim edge-CMI as a method. Pivot to node/graph diagnostic or return to CITA.

---

## Gate 3 — Regularizer effect

### Required evidence

Relative to GraphCMI-ERM:

- Full CIGL reduces graph/node/edge leakage.
- Accuracy does not collapse.
- At least one of node-CMI or edge-CMI provides benefit beyond global-CMI-only.
- Conditional penalty behaves better than marginal domain removal under label/domain imbalance.

### Quantitative pass target

At least on two datasets or one dataset plus synthetic:

- leakage reduction: ≥30% for the primary leakage object;
- accuracy loss: ≤2 balanced-accuracy points, unless worst-subject accuracy improves;
- edge or node term improves leakage–accuracy Pareto over global only.

### Fail action

Downgrade claim to diagnostic-only. Do not write “CIGL improves generalization” without evidence.

---

## Gate 4 — Strict DG validity

### Required evidence

- No target labels used in model selection.
- No target covariates used in training, normalization, or alignment.
- Source-only validation split documented.
- Random seeds logged.
- Commit hash and config hash logged.
- CITA/TTA results, if any, are clearly separated.

### Pass condition

A reviewer can reproduce which data each method used.

### Fail action

Invalidate contaminated results and rerun.

---

## Gate 5 — Paper-level evidence

### Required evidence

- At least 3 real datasets or 2 real datasets + strong synthetic sanity.
- At least 5 seeds for main comparisons.
- Main table includes task and leakage metrics.
- Ablation table includes graph/node/edge terms.
- A figure shows adjacency or node leakage as subject fingerprint.
- A reviewer-risk appendix exists.
- Scripts can rebuild main tables from result files.

### Pass condition

The paper has a coherent contribution even if CIGL does not beat every external EEG-GNN SOTA number.

### Fail action

Do not submit as a full method paper. Convert into a workshop/diagnostic paper or continue experiments.

---

## Absolute red lines

Reject any paper draft or result package that:

1. Calls CIGL transductive when main results are strict, or calls CITA strict DG.
2. Uses target data in a strict DG table.
3. Claims unbiased CMI estimation.
4. Hides raw-vs-DE feature differences.
5. Reports accuracy without leakage.
6. Reports leakage without task preservation.
7. Uses only one seed for main claims.
8. Omits node/edge ablations.
