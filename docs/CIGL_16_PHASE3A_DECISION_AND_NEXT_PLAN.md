# CIGL Phase 3A — Decision and Next Plan

This is the decision record for the Phase 3A regularizer-effect pilot
(`docs/CIGL_15_PHASE3A_RESULTS_BNCI2014_001.md`, job 876274, BNCI2014_001 fold-0). It records the
Gate-3A verdict and the next authorized step. It is a project record, not a results table.

## Gate-3A decision

- **Controllability: PASS.** CMI regularization is real and effective — graph/node terms drive the
  measured graph and node leakage from ~0.55 down to ~`1e-3`–`1e-4` (≈99.9–100% reduction), with
  significance under the retrained permutation null.
- **Task-preserving CIGL method gate: FAIL.** No tested config reduces graph/node leakage by ≥30%
  *and* keeps the source balanced-accuracy drop ≤3 points.
- **Full CIGL (Path A): NOT APPROVED** under the current baseline and λ scale.

## Why it fails (the bottleneck is baseline adequacy, not leakage control)

1. **Graph/node configs erase leakage but cost task accuracy.** All graph/node-capable configs drop
   source bAcc by 3.7–7.2 points (graph_node −7.2, full_cigl −6.3, graph_only −5.9, node_only −5.4,
   low_full_cigl −3.7). Even the gentlest tested (λ=0.1) exceeds the ≤3-pt budget — the terms act like
   a hard information eraser at this strength.
2. **`edge_only` preserves task but does not control the dominant leakage.** It keeps source bAcc
   (0.330 ≈ ERM 0.329) and removes edge leakage (98.6%), but leaves graph/node leakage ~80% intact
   (only 17–19% reduction). So the edge term alone is not a solution to the graph/node leakage.
3. **The GraphCMINet-ERM baseline is near chance.** Source bAcc ≈ 0.329 on a 4-class task (chance 0.25)
   is too close to chance to support a paper-level method verdict. The task-vs-leakage tradeoff is
   therefore **fragile**, not decisive: the "task cost" is being measured where the task signal is
   already weak.

## What this means

CIGL can control leakage. The open question is now whether a **credible task baseline** exists under
strict source-only training, and whether **gentler** CMI weights can then buy a task-preserving leakage
reduction. Until the baseline is non-degenerate, CIGL should not be judged as a training method.

## Next authorized step — Phase 3A-R (baseline repair + gentle-λ re-pilot)

> **Phase 3A-R is the only authorized next work.** Goal: (1) establish a non-degenerate
> GraphCMINet-ERM baseline on BNCI2014_001 fold-0 under source-only selection; (2) if the baseline
> passes, re-test a **gentle** CMI micro-ladder (λ ≤ ~0.05) on that fixed baseline; (3) if the baseline
> cannot be repaired, stop and diagnose architecture/preprocessing before any further method claim.

Branch: `project/cigl-phase3a-baseline-repair`. It has its own dry-run + reviewer checkpoint before any
GPU run. Decision branches afterward:

- baseline repaired **and** gentle λ gives a task-preserving reduction → revive graph+node / full CIGL;
- only edge stays task-preserving → narrow to **Edge-CMI**;
- no task-preserving tradeoff at any credible baseline → pivot to a **diagnostic framework**.

## Explicitly NOT authorized (until Phase 3A-R reports)

- Full LOSO (all folds), SEED / DEAP, or any benchmark / SOTA table.
- A large λ grid (only a small named micro-ladder is allowed).
- Phase 3B or any method-paper framing.
- Pivoting to Edge-CMI now (wait for a repaired baseline).

## Standing rule (unchanged)

Target labels may be used **only** for after-the-fact `target_eval` metrics. They must never be used
for training, early stopping, normalization, config selection, probe fitting, the leakage audit, or
model selection. Every artifact records `used_target_labels_for_training=false`,
`used_target_labels_for_selection=false`, `used_target_covariates=false`,
`target_eval_is_evaluation_only=true`.
