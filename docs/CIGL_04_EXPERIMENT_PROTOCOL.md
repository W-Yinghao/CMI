# CIGL Experiment Protocol

## 1. Evaluation setting

The CIGL main setting is **strict source-only domain generalization**.

Allowed during training and model selection:

- source examples;
- source labels;
- source domain IDs;
- source-only validation splits;
- nested source-domain selection.

Forbidden in the main table:

- target labels;
- unlabeled target covariate statistics;
- target normalization fitted on target data;
- test-time alignment;
- transductive adaptation.

CITA, CORAL/TTA, SPDIM, T3A, and other target-covariate methods must be separated into a transductive/TTA table if used at all.

---

## 2. Domain definitions

Default:

```text
D = subject
```

For cross-session:

```text
D = subject-session
```

For cross-site clinical datasets:

```text
D = site/cohort
```

Do not use `D=subject` when each subject is single-class and then interpret decoder/concept probes as meaningful. Report validity flags for domain-class span.

---

## 3. Dataset priority

### Primary

| Dataset | Role | Main protocol |
|---|---|---|
| SEED | EEG emotion, GNN natural testbed | LOSO |
| SEED-IV | harder multi-class emotion | LOSO |
| DEAP | raw-signal emotion robustness | subject-wise split / LOSO if feasible |

### Supplemental

| Dataset | Role | Main protocol |
|---|---|---|
| BNCI2014_001 | motor imagery sanity | LOSO |
| BNCI2014_004 | binary MI sanity | LOSO |
| Lee2019_MI | scale | LOSO / cross-session |
| SCPS / clinical | separate clinical section only if domain definition valid | site/cohort |

Do not overload the first paper with too many datasets before the graph/node/edge hypothesis is proven.

---

## 4. Baseline taxonomy

### 4.1 Raw-input baselines

These are the fairest main-table baselines:

1. Raw temporal stem + MLP.
2. EEGNet-ERM.
3. EEGNet + LPC-CMI.
4. GraphCMI-ERM.
5. DGCNNBackbone-ERM, raw version.
6. RGNNBackbone-ERM, raw version.

### 4.2 GNN structural baselines

Use if implementation capacity allows:

1. Shared-adjacency DGCNN-style model.
2. Shared-adjacency RGNN-style model.
3. Per-sample adjacency without edge-CMI.
4. Per-sample adjacency with only sparsity.
5. NodeDAT-style adversarial node-domain training.

### 4.3 CMI ablations

Mandatory:

1. Graph only: \(\lambda_g>0,\lambda_n=\lambda_e=0\)
2. Node only or graph+node
3. Edge only or graph+edge
4. Full CIGL
5. Marginal domain penalty instead of conditional CMI
6. Uniform prior instead of \(\pi_y(D)\)

### 4.4 Literature comparisons

When citing literature EEG-GNN numbers, label input protocols clearly:

- raw EEG;
- DE/band-power;
- subject-dependent;
- subject-independent / LOSO;
- transductive vs strict source-only.

Never present DE-feature SOTA as directly equivalent to raw-input CIGL unless we also run the same DE input pipeline.

---

## 5. Metrics

### Required task metrics

- balanced accuracy;
- macro-F1;
- worst-subject balanced accuracy;
- standard deviation across held-out subjects;
- NLL;
- ECE.

### Required leakage metrics

- held-out \(\widehat I(Z_g;D\mid Y)\);
- held-out \(\widehat I(A;D\mid Y)\);
- mean node leakage;
- max node leakage;
- node leakage map stability;
- edge leakage above permutation null;
- conditional domain-prediction advantage over \(\pi_y(D)\).

### Required representation metrics

- label separability;
- graph adjacency non-collapse statistics;
- average edge density;
- adjacency variance across samples;
- adjacency variance across subjects.

---

## 6. Model selection

Model selection must be source-only. Recommended rule:

\[
\theta^*=
\arg\min_{\theta\in\Theta}
\widehat I_{val}(Z_g;D\mid Y)
+
\alpha\widehat I_{val}(A;D\mid Y)
+
\beta\widehat I_{val}(Z_v;D\mid Y)
\]

subject to:

\[
A_{val}(\theta) \ge A_{max}-\epsilon.
\]

Default \(\epsilon\): 1–2 balanced-accuracy points.

Simpler first-version selection:

1. choose all configs within 1 point of best source-validation balanced accuracy;
2. pick lowest validation edge+node+graph leakage.

Report both accuracy-only and CMI-aware selector in ablation.

---

## 7. Statistical reporting

Minimum for real datasets:

- 5 seeds for main tables;
- per-target-subject records, not only means;
- mean ± standard error;
- paired test or bootstrap over target subjects for key comparisons;
- result JSON/CSV with commit hash and config hash.

---

## 8. Main figures and tables

### Figure 1 — Method pipeline

Raw EEG → node encoder → dynamic adjacency → graph propagation → graph/node/edge CMI.

### Figure 2 — Learned adjacency as subject fingerprint

Show edge/domain probe performance and permutation null.

### Figure 3 — Accuracy–leakage Pareto

Compare ERM, global only, node, edge, full CIGL.

### Figure 4 — Node/edge leakage map

Electrode or connectivity map showing residual subject leakage.

### Table 1 — Main results

Balanced accuracy, worst-subject accuracy, graph leakage, node leakage, edge leakage.

### Table 2 — Ablations

Conditional vs marginal, graph/node/edge components, per-sample vs shared adjacency.
