# CIGL Project Charter

**Project name:** CIGL — Conditional Information Graph Learning for Calibration-Free EEG Generalization  
**Internal alias:** GraphCMI  
**Status:** New main project, quality-first, venue-flexible.  
**Owner role model:** Manager + reviewer discipline. Every implementation step must produce evidence, not only code.

---

## 1. Project decision

We will not make CITA the new project. CITA already exists in this repository as a transductive / test-time alignment branch. It is useful, but it is not the cleanest vehicle for a new scientific contribution because its main accuracy lever is target-covariate alignment rather than conditional information learning.

The new project is **CIGL / GraphCMI**:

> Dynamic EEG graph models learn subject-dependent spatial-connectivity structure. CIGL treats the learned graph itself as a possible subject fingerprint and regularizes label-conditional domain information at graph, node, and edge levels.

The key scientific object is the per-sample learned adjacency:

\[
A(X) \in \mathbb{R}^{C \times C}.
\]

The key information-theoretic question is:

\[
I(A;D\mid Y),\quad \sum_{v=1}^{C} I(Z_v;D\mid Y),\quad I(Z_g;D\mid Y).
\]

This makes the project different from flat Tri-CMI, different from CITA/TTA, and different from standard EEG-GNN domain-adversarial methods.

---

## 2. One-line thesis

> EEG-GNNs can improve cross-subject decoding by learning dynamic spatial graphs, but the learned adjacency and node representations may encode subject identity. CIGL suppresses label-preserving domain leakage in graph embeddings, electrode-level node embeddings, and learned connectivity while preserving task information under strict source-only deployment.

---

## 3. What is new

CIGL is not just “CMI on a GNN backbone.” The novelty must be framed as three named levels:

1. **Graph-level conditional leakage**
   \[
   I(Z_g;D\mid Y)
   \]
   This is the direct extension of existing LPC-CMI.

2. **Node-level conditional leakage**
   \[
   \frac{1}{C}\sum_{v=1}^{C} I(Z_v;D\mid Y)
   \]
   This asks which electrodes carry residual subject/domain identity after conditioning on the task label.

3. **Edge-level conditional leakage**
   \[
   I(A;D\mid Y)
   \]
   This asks whether the learned connectivity pattern itself is a subject fingerprint.

The edge term is the project’s strongest distinct contribution. Without it, the project risks looking like a simple GNN host for an existing CMI regularizer.

---

## 4. Relationship to existing repository branches

| Existing repository component | CIGL role |
|---|---|
| Tri-CMI / LPC-CMI | Theoretical and estimator foundation. |
| Dual-CMI / DualPC | Historical branch and negative/diagnostic context. Not a main contribution. |
| CITA | Existing transductive/TTA branch. Use only as a separate deployment baseline or appendix. |
| GraphCMI code/design | Starting point for the new project. Must be cleaned, tested, documented, and validated. |
| H²-CMI | Future theory/simulator direction. Not part of CIGL first paper. |
| FMCA / CHSIC | Estimator robustness or negative ablation. |
| SSL-CMI | Future extension. Not in the first paper unless already strong. |

---

## 5. Target identity

We aim for an **AAAI-quality** method paper, but not at the cost of weak evidence. If the evidence is not ready for AAAI, the project should move to a later venue rather than forcing a rushed submission.

The paper must read as a self-contained new project:

- strict source-only EEG domain generalization;
- graph/node/edge conditional leakage as the central object;
- learned adjacency as subject fingerprint;
- node and edge leakage maps as diagnostic evidence;
- ablation ladder showing why graph, node, and edge terms are necessary.

---

## 6. Non-goals

Do not spend first-project capacity on these items:

1. **Do not position CIGL as CITA.** CITA uses unlabeled target covariates and belongs to transductive/TTA evaluation.
2. **Do not claim strict SOTA solely on raw accuracy.** The core claim is leakage-aware robust generalization and structural diagnosis.
3. **Do not mix raw and DE-feature results without protocol labels.** Many EEG-GNN baselines use DE/band-power features; CIGL’s default backbone is raw-signal.
4. **Do not call the posterior-KL surrogate an unbiased CMI estimator.** It is a trainable plug-in proxy / leakage objective.
5. **Do not add many auxiliary losses early.** First prove graph/node/edge conditional information contributes.
6. **Do not use target labels or target covariates in the strict DG main table.**

---

## 7. Claim boundary

Allowed claims:

- “CIGL reduces label-conditional domain leakage in graph, node, and edge representations.”
- “Learned EEG adjacency can be audited as a subject/domain fingerprint.”
- “Node and edge conditional leakage maps reveal where subject identity remains after conditioning on labels.”
- “CIGL improves the accuracy–leakage or worst-subject tradeoff over raw GraphCMI-ERM and global-CMI-only baselines.”

Disallowed claims unless explicitly proven:

- “CIGL is causal.”
- “CIGL estimates true CMI unbiasedly.”
- “CIGL always beats all EEG-GNN SOTA in absolute accuracy.”
- “CIGL is test-time adaptation.”
- “CIGL solves concept shift.”

---

## 8. Managerial rule

Every implementation step must produce one of:

- a unit test;
- a smoke result;
- a result table;
- a diagnostic artifact;
- a reviewer-risk answer.

Code without evidence does not count as progress.
