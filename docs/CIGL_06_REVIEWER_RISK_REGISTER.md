# CIGL Reviewer Risk Register

This document lists the attacks a strong reviewer will make and the evidence required before submission.

---

## Risk 1 — “This is just Tri-CMI on a GNN.”

### Reviewer concern

The method may look like an existing flat representation regularizer hosted by a graph backbone.

### Required defense

- Define three separate objects: graph embedding, node embeddings, learned adjacency.
- Show edge-CMI \(I(A;D\mid Y)\) and node-CMI \(\sum_v I(Z_v;D\mid Y)\) are not reducible to graph-level CMI.
- Include ablation: global only vs graph+node vs graph+edge vs full.
- Include edge leakage diagnostic figure.

### Claim language

“CIGL extends conditional leakage control from pooled representations to structural graph objects — especially learned adjacency — enabling domain-leakage auditing and suppression at graph, electrode, and connectivity levels.”

---

## Risk 2 — “Why not NodeDAT / adversarial training?”

### Reviewer concern

RGNN-like methods already used node-level domain adversarial training.

### Required defense

- NodeDAT is marginal/adversarial; CIGL is label-conditional and KL-to-prior.
- Show marginal removal harms class-correlated signal under label/domain imbalance.
- Show node leakage maps, which CIGL naturally produces.
- Include NodeDAT-style baseline if feasible.

### Claim language

“CIGL is a non-adversarial, label-conditional alternative to node-level domain confusion and avoids erasing task-relevant label-conditioned structure.”

---

## Risk 3 — “Why not FreqDGT / dynamic EEG-GNNs?”

### Reviewer concern

Recent methods already learn dynamic graphs and use adversarial disentanglement.

### Required defense

- Dynamic graph learning is not the same as measuring/suppressing domain information in the learned adjacency.
- CIGL asks whether the learned graph is a subject fingerprint.
- Compare conceptually and, if feasible, empirically.
- Clearly label input differences: raw EEG vs rPSD/DE/band features.

### Claim language

“Prior dynamic EEG-GNNs learn input-specific connectivity; CIGL regularizes and audits the domain information contained in that connectivity.”

---

## Risk 4 — “Raw input underperforms DE-feature GNNs.”

### Reviewer concern

External EEG-GNN SOTA often uses DE/band-power features and may have higher absolute accuracy.

### Required defense

- Use fair raw-input baselines in the main table.
- Label DE-feature literature numbers separately.
- Emphasize delta over raw GraphCMI-ERM and leakage/worst-subject improvements.
- Optionally add a DE-input appendix if available, but do not mix protocols.

### Claim language

“Our primary comparison is within a raw-input, source-only protocol; DE-feature literature numbers are reported as protocol context, not direct apples-to-apples baselines.”

---

## Risk 5 — “CMI estimator is unreliable.”

### Reviewer concern

High-dimensional CMI estimation is difficult.

### Required defense

- Do not claim unbiased estimation.
- Treat posterior-KL as a trainable leakage proxy.
- Use held-out probes, not only in-loop losses.
- Include permutation nulls.
- Include estimator robustness ablation if possible: logistic probe, MLP probe, CHSIC/FMCA/matrix-Rényi diagnostic.

### Claim language

“We optimize a posterior-KL proxy for conditional leakage and validate its behavior with held-out probes and within-label permutation tests.”

---

## Risk 6 — “Edge-CMI will collapse the graph.”

### Reviewer concern

Penalizing adjacency domain information may make the learned graph uninformative.

### Required defense

- Track adjacency density, variance, and label separability.
- Warm up edge-CMI.
- Show edge-CMI reduces domain leakage without destroying task edges in synthetic data.
- Include graph-only and sparsity-only ablations.

### Claim language

“CIGL does not minimize graph information globally; it suppresses label-conditional domain information while preserving task loss.”

---

## Risk 7 — “This is actually CITA/TTA.”

### Reviewer concern

The repository contains transductive alignment code, and the paper may confuse settings.

### Required defense

- Main results: no target labels, no target covariates.
- CITA separated into appendix or omitted.
- Experiment table has a `setting` column.
- README and docs explicitly distinguish CIGL and CITA.

### Claim language

“CIGL is a source-only representation-learning method. CITA is a separate transductive deployment branch and is not used in the strict DG main results.”

---

## Risk 8 — “Why conditional on Y?”

### Reviewer concern

A reviewer may ask why not remove all domain information marginally.

### Required defense

- Synthetic label/domain imbalance experiment.
- Marginal domain removal baseline.
- Show marginal invariance can erase class-correlated structure.
- Show conditional CMI reduces domain leakage within each label without forcing label-prior matching.

### Claim language

“EEG domain and label distributions are often entangled. Conditioning on \(Y\) targets domain leakage that remains after the task label is known, reducing label-erasure risk.”

---

## Risk 9 — “Node/edge maps are not stable.”

### Reviewer concern

Interpretability artifacts may be seed noise.

### Required defense

- Report rank correlation of node maps across seeds.
- Bootstrap top-k channel stability.
- For edge maps, report region-level aggregation if per-edge maps are too noisy.
- Do not over-interpret individual edges without stability.

### Claim language

“We use leakage maps as diagnostic summaries and report stability; unstable maps are not interpreted neurophysiologically.”

---

## Risk 10 — “Method stacking.”

### Reviewer concern

The method may appear to add many losses until something works.

### Required defense

- Keep first method minimal: CE + graph/node/edge CMI.
- No SSL, no CITA, no H²-CMI, no extra alignment in the main method.
- Each loss term has a named object and ablation.

### Claim language

“Each regularizer corresponds to a different graph object: pooled graph embedding, electrode node features, and learned adjacency.”
