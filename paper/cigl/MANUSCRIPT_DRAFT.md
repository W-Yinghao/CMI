# CIGL: Auditing and Reducing Label-Conditional Domain Leakage in EEG Graph Representations under Source-Only Generalization

> **DRAFT (Phase 4B).** Bounded first-paper skeleton. Citations are author-named where known and marked
> `TODO: verify citation` until details are checked; no BibTeX is invented. Numbers trace to
> `docs/CIGL_25/29/31`. Language is deliberately cautious (see `CLAIMS_AUDIT.md`).

## Abstract (draft)

Cross-subject electroencephalography (EEG) decoders are confounded by subject- and session-specific
structure that a model can exploit instead of the task signal. We study this as **label-conditional
domain leakage** in *learned EEG graph representations*: how much subject/domain identity a graph network's
graph-level and node-level features carry once the class label is fixed. We measure it with a
**posterior-KL plug-in proxy** — a conditional-domain probe scored against a retrained, within-label
permutation null — and emphasize that this proxy is *not* an unbiased conditional-mutual-information (CMI)
estimator. On a task-capable graph backbone (a DGCNN with a shared/static adjacency), we find the graph
and node representations carry significant such leakage. We then add a fixed **graph/node** conditional-
information regularizer (no edge term) and show, under **strict source-only** domain generalization on two
motor-imagery (MI) datasets, that it **partially but reproducibly reduces** this leakage **without harming
source-task performance**. The reduction is partial (the regularized leakage still clears the null), the
estimator is a proxy, the backbone is a single static-adjacency graph network, and the evidence is two MI
datasets — we make no SOTA, leakage-elimination, edge-CMI, cross-architecture, or beyond-MI claim. We also
report the negative results that shaped the method (a near-chance graph backbone and overfitting dynamic-
edge designs), which justify the bounded scope.

## 1. Introduction

EEG brain–computer interfaces must generalize across subjects, yet inter-subject variability is large and
a decoder can latch onto subject identity rather than the neural task. A growing line of work treats this
as a domain-generalization (DG) problem and as a *shortcut/leakage* problem: features that separate
subjects, conditioned on the label, are a direct measure of how much the representation could be exploiting
domain identity. Graph neural networks are attractive for EEG because electrodes form a natural graph, and
they expose interpretable objects — a graph-level readout, per-electrode node features, and (for some
designs) a learned adjacency. This makes them a good substrate for *auditing* leakage at the graph and node
level.

This paper makes a deliberately **bounded** contribution. First, we define and validate a source-only
**audit** of label-conditional domain leakage in a graph network's `graph_z` and `node_z` objects, using a
posterior-KL proxy with a retrained within-label permutation null and a strict no-target-label firewall.
Second, we report a methodology chain — including its negative results — that establishes a key
prerequisite: leakage control is only meaningful on a *task-capable* graph backbone, and identifying one is
nontrivial. Third, we introduce a fixed graph/node conditional-information regularizer and show, under
strict source-only DG on two MI datasets, that it partially and reproducibly reduces the audited leakage at
no task cost. We do not claim state-of-the-art accuracy, leakage elimination, an unbiased CMI estimate, an
edge-level method, or generality beyond this backbone and these two MI datasets.

## 2. Related Work

*(See `RELATED_WORK_MATRIX.md` for the positioning grid; citations carry `TODO: verify citation`.)*
We connect to four threads: (i) **EEG domain generalization / cross-subject MI**, especially the MOABB
benchmarking protocol [TODO: verify citation] from which our datasets and LOSO evaluation derive; (ii)
**graph EEG networks** — DGCNN [TODO: verify citation], RGNN [TODO: verify citation], and LGGNet [TODO:
verify citation] — which motivate graph/node/edge objects; (iii) **known-good convolutional MI decoders**
— EEGNet and ShallowConvNet/DeepConvNet [TODO: verify citation] — which we use only as sanity references
for task learnability; and (iv) **conditional-invariance / information-penalty** methods (conditional
domain-invariant representations, classifier-based CMI / posterior-KL leakage proxies, and domain-
adversarial or marginal-invariance baselines) [TODO: verify citation]. CIGL differs by (a) *auditing* graph
and node leakage explicitly with a permutation-null proxy before regularizing, (b) operating strictly
source-only with target labels evaluation-only, and (c) restricting to a graph/node penalty on a
task-capable static-adjacency backbone rather than claiming a general edge/dynamic-graph method.

## 3. Method

**Backbone.** We use a DGCNN-style graph network with a single shared (static) learned adjacency and a
Chebyshev graph convolution over electrode nodes, exposing `forward_graph(x) → (logits, graph_z, node_z,
edge_logits)`. Because the adjacency is shared across samples, there is **no per-sample edge object**
(`edge_logits = None`); CIGL as studied here is therefore **graph/node only**. We adopt this backbone after
an explicit search (Section 6): it is the graph-compatible backbone we found that actually learns the task.

**Leakage objects and proxy.** Let `Z_g` be the graph readout and `Z_v` the per-electrode node features,
`Y` the class label, and `D` the source-domain (subject) label. We quantify label-conditional domain
leakage with a **posterior-KL plug-in proxy**:
`R_g = E[ KL( q_g(D | Z_g, Y) ‖ π_y(D) ) ]` and `R_n = (1/C) Σ_v E[ KL( q_n(D | Z_v, v, Y) ‖ π_y(D) ) ]`,
where `q_g`/`q_n` are conditional-domain posterior heads, `π_y(D)` the within-label domain prior, and `C`
the number of electrodes. We stress that `R_g`/`R_n` are a **proxy**, not an unbiased CMI estimator.

**Regularizer.** CIGL trains the backbone with `L = L_CE + λ_g R_g + λ_n R_n` (no edge term), using the
established two-step scheme: Step A fits the posterior heads on detached features; Step B penalizes the
encoder. Throughout the confirmation we use a single **fixed** weight `λ_g = λ_n = 0.010` (`λ_edge = 0`).
[FIGURE: CIGL pipeline schematic — F1.]

**Audit.** To judge significance we re-fit fresh held-out conditional-domain probes on frozen features and
compare to a **retrained, within-label permutation null** (permuting `D` within label on the probe-training
split only, preserving the within-label prior); `clears_null = kl_mean > permutation_mean AND
permutation_p ≤ 0.05`. The audit is the same proxy used as a measurement, never inverted into an
information-theoretic guarantee. [FIGURE: graph/node leakage audit schematic — F3.]

## 4. Experimental Protocol

We evaluate under **strict source-only DG**: a leave-one-subject-out (LOSO) target is excluded from
training and from the source probe used for selection; **target labels are evaluation-only** and never used
for training, early stopping, normalization, model/config selection, confirmation-label choice, probe
fitting, or the audit. We use two MI datasets: **BNCI2014_001** (4-class, chance 0.25) and **BNCI2015_001**
(binary right_hand vs feet, chance 0.50; loaded via the `MotorImagery` paradigm because it is not
left/right-hand). Both use MOABB preprocessing [TODO: verify citation] at 128 Hz, a 0.5–3.5 s window, and
per-trial z-scoring; data come from a read-only datalake. The candidate `λ_g=λ_n=0.010` was selected once
on a single development fold (BNCI2014_001 fold-0) and then **frozen**; all subsequent runs are
fixed-candidate confirmations with no λ search. [TABLE: method/config/protocol — T1.]

## 5. Results

**Leakage exists on a task-capable backbone.** On the DGCNN backbone (BNCI2014_001 fold-0), the graph and
node objects carry strong, significant, and spatially stable leakage: graph proxy ≈ 8× and node proxy ≈ 15×
the permutation mean, clearing the null in 3/3 seeds, with a node-leakage map highly reproducible across
seeds. This is the first such audit on a backbone that *also* learns the task. [TABLE: DGCNN leakage audit
— T2.]

**Fixed regularizer reduces leakage at task retention — two datasets.** On **BNCI2014_001** (primary folds
1–8; fold-0 is the development fold and is excluded), the fixed `graph_node_010` passes all four source-only
criteria in every primary fold, reducing graph proxy by ≈ 35–58% and node proxy by ≈ 31–45% while retaining
source accuracy (absolute drop ≤ 0.02). [TABLE: BNCI2014_001 confirmation — T3.] On **BNCI2015_001** (all
12 LOSO folds), the same fixed candidate is ERM-adequate, leakage-bearing, and leakage-reducing in 12/12
folds, retains source accuracy in 11/12 folds, and holds the evaluation-only target guardrail in 12/12
folds; graph proxy drops ≈ 43–77% and node proxy ≈ 37–61%, yielding `confirmed_with_target_guardrail =
true`. [TABLE: BNCI2015_001 confirmation — T4.] [FIGURE: leakage-reduction vs task-retention scatter — F2.]

**Partial, not total.** In every confirmation fold the *regularized* leakage still clears the null — CIGL
reduces (≈ 40–65%) but does not eliminate the leakage. We report reduction, not removal.

## 6. Analysis and Negative Results

The negative results are part of the contribution; they justify the bounded scope. (i) **The original
graph backbone fails the task.** A GraphCMINet-style network is near-chance under strict source-only
training and is not repairable by small training tweaks, so its (strong, controllable) leakage cannot
support a task-vs-leakage tradeoff. (ii) **The protocol is learnable.** Known-good convolutional decoders
(EEGNet/ShallowConvNet/DeepConvNet) and a DGCNN all clear a source-accuracy floor on the same fold, showing
the bottleneck was the specific graph backbone, not the data/protocol. (iii) **Dynamic-edge designs
overfit.** Graph backbones with a per-sample (dynamic) adjacency memorize the source (train ≈ 1.0, source
≈ chance) regardless of temporal stem, so a dynamic per-sample edge object — the `I(A;D|Y)` "fingerprint" —
is currently task-harmful here, and **edge-CMI is out of scope**. The static-adjacency DGCNN is the only
graph-compatible backbone that both learns the task and exposes usable graph/node objects, which is why
CIGL is framed as graph/node only on this backbone. [TABLE: negative-results summary — T5.] [FIGURE:
negative-result decision flow — F4.]

## 7. Limitations

(1) The leakage metric is a **posterior-KL proxy**, not an unbiased CMI estimator; it can over- or
under-state true CMI, so we frame it as an audit/penalty, not a guarantee. (2) Control is **partial** —
the regularized leakage still clears the null. (3) Evidence is **two MI datasets**; we make no beyond-MI or
universal claim. (4) Results are for **one backbone** (DGCNN static adjacency); no cross-architecture claim.
(5) A **single fixed λ** (0.010) was confirmed; we ran no λ-grid and make no λ-robustness claim. (6) The
candidate was selected on a development fold before freezing; we report this and exclude that fold from the
primary aggregate. (7) Baselines are modest (BNCI2014_001 ERM ≈ 0.46; BNCI2015_001 ERM ≈ 0.70) — this is a
leakage-reduction-at-retention result, not an accuracy record. (8) **No edge object** (static adjacency);
edge/dynamic-graph leakage control is future work.

## 8. Conclusion

We presented CIGL, a source-only audit and graph/node regularizer for label-conditional domain leakage in
EEG graph representations. On a task-capable static-adjacency DGCNN backbone, the audited graph/node leakage
is significant, and a fixed graph/node conditional-information regularizer partially and reproducibly
reduces it without harming source-task performance on two MI datasets. The contribution is a bounded,
honestly-scoped measurement→control result, accompanied by the negative results that shaped it. Future work
includes constrained dynamic-edge backbones that generalize (to enable an edge-level audit), additional
datasets and paradigms, and tighter estimators than the posterior-KL proxy.
