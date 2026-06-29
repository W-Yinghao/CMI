# CIGL Paper Blueprint

This is a quality-first paper plan. Venue choice is made only after Gate 5 evidence is available.

---

## 1. Candidate titles

1. **CIGL: Conditional Information Graph Learning for Calibration-Free EEG Generalization**
2. **Is the Learned EEG Graph a Subject Fingerprint? Conditional Information Regularization for EEG Domain Generalization**
3. **Graph, Node, and Edge Conditional Leakage Control for Source-Only EEG Generalization**

Preferred title:

> **Is the Learned EEG Graph a Subject Fingerprint? Conditional Information Graph Learning for EEG Generalization**

This title foregrounds the strongest scientific question.

---

## 2. Abstract skeleton

Cross-subject EEG decoding remains difficult because learned representations often encode subject-specific nuisance structure. While graph neural networks are natural for EEG spatial modeling, their learned connectivity can itself become a subject fingerprint. We introduce CIGL, a source-only framework that regularizes label-conditional domain information at three granularities: pooled graph embeddings, electrode-level node embeddings, and per-sample learned adjacency. CIGL uses posterior-KL conditional leakage proxies to suppress \(I(Z_g;D\mid Y)\), \(\sum_v I(Z_v;D\mid Y)\), and \(I(A;D\mid Y)\) while preserving task prediction. Across strict source-only EEG generalization benchmarks, CIGL provides leakage-aware graph learning, node/edge leakage maps, and improved accuracy–leakage or worst-subject tradeoffs over raw graph and global-CMI baselines. Our results show that auditing learned connectivity is critical for robust EEG graph learning.

Adjust “improved” only after evidence is available.

---

## 3. Main contributions

Use exactly four contributions:

1. **New object:** We identify per-sample learned EEG adjacency as a conditional domain-leakage object and evaluate whether it acts as a subject fingerprint.
2. **Method:** We introduce graph-, node-, and edge-level posterior-KL CMI regularization for strict source-only EEG DG.
3. **Diagnostics:** We produce node and edge leakage maps that localize residual subject/domain information after conditioning on the task label.
4. **Evidence:** We evaluate CIGL under strict source-only protocols with ablations against global-only CMI, marginal domain removal, raw GNN baselines, and adversarial node-domain training.

---

## 4. Paper structure

### 1. Introduction

- EEG DG problem.
- GNNs model spatial connectivity.
- Learned connectivity can encode subject identity.
- Flat representation invariance misses node/edge structure.
- CIGL contribution.

### 2. Related Work

Organize by problem, not by long survey:

1. EEG domain generalization.
2. EEG graph neural networks.
3. Domain-adversarial and invariant representation learning.
4. Information-theoretic graph learning.
5. Conditional mutual information and leakage diagnostics.

### 3. Method

3.1 Source-only EEG DG setup.  
3.2 GraphCMINet backbone.  
3.3 Posterior-KL conditional leakage proxy.  
3.4 Graph/node/edge CIGL losses.  
3.5 Training and model selection.  
3.6 Diagnostics and leakage maps.

### 4. Experiments

4.1 Protocol and datasets.  
4.2 Baselines.  
4.3 Metrics.  
4.4 Implementation details.

### 5. Results

5.1 Does learned adjacency encode subject identity?  
5.2 Main strict DG results.  
5.3 Accuracy–leakage Pareto.  
5.4 Node/edge leakage maps.  
5.5 Ablations.

### 6. Limitations

- CMI proxy is not unbiased.
- Maps are diagnostic, not definitive neurophysiology.
- Raw-vs-DE protocol differences.
- Edge maps may be noisy for small datasets.
- CITA/TTA is a separate deployment setting.

### 7. Conclusion

Return to the central claim: learned EEG graphs are useful but can leak subject identity; conditional information control gives a principled way to audit and reduce that leakage.

---

## 5. Required figures

### Figure 1 — CIGL pipeline

Must show three regularization arrows:

- graph \(Z_g\);
- nodes \(Z_v\);
- adjacency \(A(X)\).

### Figure 2 — Adjacency as subject fingerprint

Use probe accuracy / KL and permutation null.

### Figure 3 — Accuracy–leakage Pareto

Methods:

- GraphCMI-ERM;
- global-CMI;
- node-CMI;
- edge-CMI;
- full-CIGL;
- marginal baseline.

### Figure 4 — Node / edge leakage map

Use stable aggregate, not one cherry-picked seed.

---

## 6. Required tables

### Table 1 — Main strict DG results

Columns:

```text
Dataset | Method | bAcc | Worst bAcc | Macro-F1 | Graph KL | Node KL | Edge KL
```

### Table 2 — Ablation

Rows:

```text
ERM
+ global CMI
+ node CMI
+ edge CMI
+ graph+node
+ graph+edge
+ full CIGL
marginal domain removal
uniform prior
shared adjacency
```

### Table 3 — Protocol comparison / appendix

Optional table separating strict DG from transductive/TTA methods.

---

## 7. Claim templates

Safe:

> CIGL consistently reduces conditional graph/node/edge leakage while maintaining task performance, improving the leakage–accuracy Pareto over global-only CMI.

Conditional on results:

> CIGL improves worst-subject balanced accuracy on [datasets], suggesting that suppressing structural leakage benefits difficult target subjects.

Unsafe unless proven:

> CIGL achieves state-of-the-art cross-subject EEG performance.

---

## 8. Reviewer-facing contribution sentence

Use this in introduction:

> Unlike prior EEG-GNNs that learn adaptive connectivity but do not constrain the domain information carried by that connectivity, CIGL explicitly treats the learned adjacency as a random representation object and suppresses its label-conditional domain leakage.
