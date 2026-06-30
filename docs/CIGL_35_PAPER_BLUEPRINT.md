# CIGL_35 — Paper Blueprint

> Phase 4A consolidation (docs only). A first-paper plan at the **bounded** scope the evidence supports.
> Not a commitment to submit; a scaffold that keeps claims honest.

## Tentative title

*Conditional Information Graph Learning: Auditing and Reducing Label-Conditional Domain Leakage in
EEG Graph Representations under Source-Only Generalization.*

(Alt: "Task-Preserving Reduction of Subject Leakage in EEG Graph Networks via Conditional Information
Regularization.")

## Abstract skeleton

- **Problem:** learned EEG graph representations can encode subject/domain identity conditional on the
  label — a leakage that confounds cross-subject generalization.
- **Audit:** a posterior-KL plug-in proxy with a retrained within-label permutation null measures
  graph- and node-level label-conditional domain leakage on a *task-capable* graph backbone.
- **Finding:** such leakage is significant on a task-capable DGCNN backbone, and a fixed graph/node
  conditional-information regularizer (`λ_g=λ_n=0.010`, no edge term) **partially but reproducibly
  reduces** it **without harming source-task accuracy**, confirmed on **two** motor-imagery datasets
  under strict source-only training.
- **Scope/honesty:** partial reduction (not elimination), posterior-KL proxy (not unbiased CMI),
  graph/node only (no edge-CMI), one backbone, two MI datasets; not a SOTA claim. Negative results
  (GraphCMINet failure, dynamic-edge overfitting) are reported.

## Contribution bullets

1. A **source-only leakage audit** for learned EEG graph objects (graph readout + per-electrode nodes),
   with a correctness-checked permutation null and a no-target-label firewall.
2. The empirical chain establishing that a **task-capable graph backbone** is a prerequisite (GraphCMINet
   fails; static DGCNN succeeds; dynamic-edge designs overfit) — a reusable methodology, not just a result.
3. A **fixed-candidate, cross-dataset confirmation** that graph/node CMI regularization reduces the
   audited leakage at task retention on two MI datasets.
4. An explicit **negative-results + limitations** account (edge-CMI unsupported here; partial not full
   control; proxy not unbiased CMI).

## Section-by-section outline

1. **Introduction** — leakage as a DG confound; the measurement→control gap; bounded contribution.
2. **Related Work** — EEG-DG (MOABB protocols), graph EEG nets (DGCNN/RGNN/LGGNet), invariance/CMI
   penalties, leakage/shortcut auditing.
3. **Method** — graph backbone; `R_g`/`R_n` posterior-KL proxies; the source-only training loop
   (Step-A posterior fit on detached features, Step-B encoder penalty); audit + permutation null.
4. **Experimental Protocol** — datasets (BNCI2014_001 4-class, BNCI2015_001 binary), LOSO, seeds, n_perm,
   fixed λ, preregistered floors/criteria, source-only firewall, datalake provenance.
5. **Results** — DGCNN leakage audit (3A-H); pilot (3A-I); BNCI2014_001 confirmation (3A-J); BNCI2015_001
   confirmation (3A-K). Per-fold reduction + task-retention tables.
6. **Analysis and Negative Results** *(mandatory)* — GraphCMINet near-chance baseline (3A-R); known-good
   decoder sanity (3A-S); dynamic-edge overfitting and why edge-CMI is unsupported (3A-G); partial (not
   full) controllability (leakage still clears the null).
7. **Limitations** — proxy vs unbiased CMI; two MI datasets; one backbone; one fixed λ; static adjacency
   only; modest baselines; no claim beyond MI.
8. **Conclusion** — a reproducible, bounded, task-preserving leakage-reduction result; future work
   (constrained dynamic-edge backbone, more datasets, tighter estimators).

## Table / figure plan

- **T1** method/config (backbone, loss, λ, setting). **T2** datasets/protocol.
- **T3** DGCNN leakage audit (graph/node KL vs perm, p, node-map stability) — 3A-H.
- **T4** BNCI2014_001 per-fold (ERM vs reg src; graph/node KL + reduction; clears; retention) — 3A-J.
- **T5** BNCI2015_001 per-fold (same columns) — 3A-K.
- **T6** negative-results summary (3A-R/3A-S/3A-G one-liners).
- **F1** pipeline schematic. **F2** per-fold reduction-vs-retention scatter (both datasets). **F3**
  node-leakage map (electrodes). **F4** ERM-vs-reg leakage bars per dataset.

## Claims allowed (verbatim-safe)

- "Significant graph/node label-conditional domain leakage on a task-capable EEG graph backbone."
- "A fixed graph/node CMI regularizer partially and reproducibly reduces this leakage without harming
  source-task accuracy, on two MI datasets, source-only."
- "Partial reduction (~40–65%), not elimination; posterior-KL proxy, not unbiased CMI."

## Claims forbidden

- Any SOTA / best-accuracy / leaderboard claim.
- "Removes/eliminates" leakage; "unbiased CMI"; "information-theoretic guarantee."
- Edge-CMI / dynamic-edge method claims.
- Cross-architecture or beyond-MI generality.
- "λ-robust" / "tuned" (no λ-grid was run).

## Negative-results section plan (required)

Frame GraphCMINet failure, dynamic-edge overfitting, and edge-CMI being unsupported as **method-shaping
evidence** (they justify the backbone choice and the graph/node-only scope), not as omissions. Report the
near-chance GraphCMINet numbers and the dynamic-edge train≈1.0/source≈chance gap explicitly.

## Limitations section plan (required)

Enumerate: proxy estimator; partial control; one backbone (static adjacency); one fixed λ; two MI
datasets; modest baselines; selection on fold-0 (dev) before freezing; no edge object; no beyond-MI claim.
