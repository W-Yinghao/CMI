# CIGL_32 — Method Framing and Claims

> Phase 4A consolidation (docs only, no new experiments). Defines the final method identity and the
> **exact** claims the evidence supports, in bounded language.

## Method identity

**CIGL — Conditional Information Graph Learning for Source-Only EEG Domain Generalization.**

A task-capable EEG **graph** backbone trained with **graph-level + node-level** conditional
information regularization that penalizes label-conditional **domain** leakage in the learned graph
objects, under a **strict source-only** domain-generalization setting.

- **Backbone:** the DGCNN static-(shared-)adjacency adapter (`dgcnn_forward_graph_adapter` in
  `cmi/models/graph_task_backbones.py`), which exposes `forward_graph(x) → (logits, graph_z, node_z,
  edge_logits=None)`. It is the *only* graph-compatible backbone found to learn the task (Phase 3A-G);
  GraphCMINet does not (Phase 3A-R/3A-S).
- **No edge term.** DGCNN's adjacency is static, so there is **no per-sample edge object**;
  `edge_logits=None`, `edge_audit_skipped=true`. CIGL as framed here is **graph/node only**.

## Loss

```
L = L_CE
  + λ_g  · R_g(Z_g ; D | Y)
  + λ_n  · R_n(Z_v ; D | Y)
                                     (λ_edge = 0 — no edge term)
```

with the **fixed** weights used throughout the confirmation:

```
λ_g = 0.010,   λ_n = 0.010,   λ_edge = 0.000          (config "graph_node_010")
```

The two regularizers are **posterior-KL plug-in proxies** for label-conditional domain leakage:

```
R_g = E[ KL( q_g(D | Z_g, Y) ‖ π_y(D) ) ]
R_n = (1/C) Σ_v  E[ KL( q_n(D | Z_v, v, Y) ‖ π_y(D) ) ]
```

where `q_g`/`q_n` are conditional-domain posterior heads fit on the (detached, Step-A) features, `π_y(D)`
is the within-label domain prior, `Z_g` is the graph readout, `Z_v` the per-node (per-electrode)
features, and `D` the source-domain (subject) label. Training matches the `graphcmi` branch of
`cmi/train/trainer.py` (Step-A fit posteriors on detached features; Step-B penalize the encoder), with
the edge term skipped when `edge_logits is None`.

> **Estimator language (use verbatim):** these are a **posterior-KL plug-in proxy for label-conditional
> domain leakage** — *not* an unbiased conditional-mutual-information (CMI) estimator. The audit metric
> (`audit_graph_node_objects`) is the same proxy with a within-label, retrained permutation null.

## Setting

Strict **source-only domain generalization**: the target subject is excluded from training and from the
`source_probe` used for selection. **Target labels are evaluation-only**; they are never used for
training, early stopping, normalization, config selection, confirmation-label choice, probe fitting, or
the audit (`used_target_labels_for_{training,selection}=false`, `used_target_covariates=false`,
`selection_uses_target_eval=false`). The candidate `graph_node_010` was selected once (Phase 3A-I,
BNCI2014_001 fold-0) and then **frozen** for all confirmation.

## What is claimed (supported by the evidence)

1. Learned EEG **graph and node** representations in a task-capable graph backbone carry **significant**
   label-conditional domain leakage (posterior-KL proxy clears a retrained within-label permutation null;
   Phase 3A-H, p at the n_perm floor, graph KL ≈ 8× / node ≈ 15× the permutation mean).
2. Graph/node CMI regularization (`graph_node_010`) **partially but reproducibly reduces** that leakage
   **without harming source-task performance**, on **two** MI datasets:
   - BNCI2014_001 (4-class): folds-1–8 primary confirmation 8/8 on every criterion; graph KL −35–58%,
     node −31–45% (Phase 3A-J).
   - BNCI2015_001 (binary): 12/12 ERM-adequate + leakage + reduction + target-guardrail, source retained
     11/12; graph KL −43–77%, node −37–61% (Phase 3A-K).
3. The effect is **task-preserving**: source `source_probe` mean stays above the preregistered floor with
   absolute drop ≤ 0.02 in the large majority of folds, and the evaluation-only target guardrail (drop
   ≤ 0.05) holds.

## What is NOT claimed (out of scope / explicitly disclaimed)

- **Not** leakage **elimination** — only **partial** reduction: the regularized leakage **still clears the
  null in every confirmation fold** (~40–65% reduction, the subject fingerprint is dented, not removed).
- **Not** an unbiased CMI estimate — a posterior-KL plug-in proxy.
- **Not** edge-CMI / dynamic-edge CIGL — the dynamic-edge backbones overfit (Phase 3A-G) and DGCNN has no
  per-sample edge object; no edge claim is made.
- **Not** SOTA / leaderboard accuracy — the claim is leakage reduction at task retention, not a new
  accuracy record (baselines are modest: 2a ERM ≈ 0.46, 2015 ERM ≈ 0.70).
- **Not** cross-architecture generality — results are for the DGCNN static-adjacency backbone only.
- **Not** large-λ robustness — a single fixed λ (0.010) was confirmed; no λ-grid was run.
- **Not** beyond motor imagery — two MI datasets only; no SEED/DEAP/other-paradigm claim.

See `CIGL_33` (evidence index), `CIGL_34` (reviewer risks), `CIGL_35` (paper blueprint), `CIGL_36`
(reproducibility).
