# CIGL Implementation Plan

This plan is ordered by evidence gates. Do not expand experiments before the previous gate is passed.

---

## Phase 0 — Projectization

### Goal

Make CIGL a clean active project in the repository.

### Tasks

1. Create `docs/` if missing.
2. Add all `CIGL_*.md` documents.
3. Add a README banner:

```md
## Active project: CIGL / GraphCMI
CIGL studies label-conditional domain leakage in EEG graph representations at graph, node, and edge levels. CITA remains a separate transductive/TTA branch.
```

4. Create `results/cigl/README.md` and `results/cigl/schema.md`.
5. Create `scripts/smoke_graphcmi.py`.
6. Create tests for the existing GraphCMI implementation.

### Acceptance

- The repository can be understood by a new reviewer without reading old Tri-CMI notes.
- CIGL, CITA, Tri-CMI, and DualPC have distinct roles.
- No scientific claim depends on future experiments.

---

## Phase 1 — Backbone and regularizer hardening

### Goal

Confirm the existing GraphCMI code is numerically valid and interface-stable.

### Tasks

1. Test `GraphCMINet` at channel counts: 19, 22, 32, 62.
2. Test `forward` and `forward_graph` contracts.
3. Test `NodePosterior` and `EdgePosterior`:
   - finite Step-A loss;
   - finite regularization loss;
   - gradient reaches `node_Z` and `edge_logits`;
   - prior tensor shape is correct.
4. Test one mini training loop with `method="graphcmi"`.
5. Add trainer failure for `method="graphcmi"` with non-graph backbone.
6. Add loss breakdown logging.

### Acceptance

- CPU smoke passes.
- No NaNs in logits, graph_Z, node_Z, edge_logits, losses.
- `edge_logits` has the intended symmetry property or documented asymmetry.
- `graphcmi` with all CMI weights zero runs as GraphCMI-ERM.
- Existing non-graph methods still run.

---

## Phase 2 — Probe-only leakage hypothesis

### Goal

Prove the core phenomenon before claiming a method: learned node/edge graph objects contain label-conditional subject/domain leakage.

### Tasks

1. Train GraphCMI-ERM on at least one small source-only split.
2. Freeze the backbone.
3. Fit held-out graph, node, and edge leakage probes.
4. Run within-label domain permutation null.
5. Save node leakage maps and edge leakage summaries.
6. Repeat across seeds.

### Acceptance

- `edge_logits` predicts domain above label-only prior under at least one dataset/protocol.
- \(I(A;D\mid Y)\) is above permutation null.
- Node leakage maps are not pure noise across seeds.
- The result can be visualized as: “the graph itself is a subject fingerprint.”

If this gate fails, the project must pivot from “regularization method” to “diagnostic framework” or return to CITA.

---

## Phase 3 — Synthetic graph sanity check

### Goal

Create a controlled DGP where the correct behavior of edge-CMI is known.

### DGP requirements

Construct synthetic graph-EEG tensors with:

1. label-relevant node/edge structure;
2. subject-spurious node/edge structure;
3. class-conditioned domain shift;
4. optional label imbalance;
5. a single-class-domain trap to expose marginal alignment failure.

### Methods

- ERM
- marginal domain adversarial / marginal leakage penalty
- global CMI only
- node CMI only
- edge CMI only
- full CIGL

### Acceptance

- Edge-CMI suppresses the known subject-spurious edge.
- It preserves the known label-relevant edge better than marginal domain removal.
- The estimated edge leakage is monotonic with injected spurious-edge strength.
- Full CIGL does not erase task signal.

Synthetic failure means the real-data method is not trustworthy.

---

## Phase 4 — Real benchmark core

### Goal

Evaluate CIGL under strict source-only EEG DG.

### Primary datasets

1. SEED — cross-subject emotion recognition.
2. SEED-IV — cross-subject emotion recognition.
3. DEAP — valence/arousal/quadrant, depending on existing loader readiness.

### Supplemental datasets

1. BNCI2014_001 / BNCI2014_004 — motor imagery.
2. Lee2019_MI — larger-scale motor imagery.
3. Clinical SCPS only as a separate section if domain definition is not degenerate.

### Methods

- Raw temporal ERM baseline.
- EEGNet-ERM.
- EEGNet + LPC-CMI.
- DGCNNBackbone-ERM.
- RGNNBackbone-ERM.
- GraphCMI-ERM.
- GraphCMI + global CMI.
- GraphCMI + node CMI.
- GraphCMI + edge CMI.
- Full CIGL.
- NodeDAT-style adversarial node baseline.
- Marginal domain adversarial or DANN/CDANN baseline.

### Acceptance

- Main table includes balanced accuracy and worst-subject accuracy.
- Main table includes graph/node/edge leakage.
- Full CIGL is Pareto-better than GraphCMI-ERM or global-CMI-only on at least two datasets, measured by accuracy/leakage or worst-subject/leakage.
- If absolute accuracy does not improve, the paper claim shifts to “leakage-control and diagnostic generalization,” not SOTA.

---

## Phase 5 — Ablation and reviewer-defense experiments

### Required ablations

1. **Conditional vs marginal:** replace \(I(\cdot;D\mid Y)\) with marginal \(I(\cdot;D)\).
2. **Graph vs node vs edge:** isolate all three terms.
3. **Per-sample adjacency vs shared adjacency:** show edge-CMI needs a per-sample graph.
4. **Raw vs DE protocol label:** do not directly compare as if identical.
5. **Prior target:** empirical \(\pi_y(D)\) vs uniform prior.
6. **Edge compression:** upper-triangle summary vs alternative summary.
7. **Warmup sensitivity:** edge warmup and lambda sweeps.
8. **Estimator reliability:** held-out probe and permutation null.

### Acceptance

- Every major reviewer objection has either a main-text result or appendix result.
- No result table silently combines strict DG and transductive TTA.
- Every loss component has a corresponding ablation.

---

## Phase 6 — Paper package

### Required paper artifacts

1. Method figure: raw EEG → node encoder → dynamic adjacency → graph propagation → graph/node/edge CMI.
2. Diagnostic figure: learned adjacency as subject fingerprint.
3. Leakage map: electrodes and/or edges with residual conditional leakage.
4. Main result table: accuracy + leakage + worst-subject.
5. Ablation table: graph/node/edge/conditional/marginal.
6. Reviewer-risk appendix.
7. Reproducibility checklist and exact commands.

### Acceptance

- The paper can be submitted as a self-contained project without depending on CITA.
- CITA can appear only as a transductive appendix/baseline.
- Claims are narrower than evidence, never broader.
