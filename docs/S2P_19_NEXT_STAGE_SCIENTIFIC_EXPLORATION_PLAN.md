# S2P_19 - Next-Stage Scientific Exploration Plan

**Status:** FROZEN scientific control plane. Project OPEN. Scientific exploration ACTIVE. Manuscript writing OFF.

This document supersedes any implication in S2P_17/S2P_18 that the project is ready to close or transition to
writing. Phase A is complete, and S2P_17 now contains the authoritative immutable through-2000 FACED result.

## Scientific objective

> Determine the order and causal source of subject-identifiable and task-transferable structure in EEG foundation
> pretraining: whether transfer emerges from more unique data, more optimization, supervised fine-tuning, or their
> combination, and whether transfer remains independent of the measured subject subspace while subject encoding
> stays strong.

The FSR evidence ladder remains the mechanism framework:

```text
subject information encoded
!= functionally used
!= target-harmful
!= safely erasable
!= repairable
```

## Current authority boundary

Load-bearing S2P evidence is limited to:

```text
CBraMod Route B
33-channel pinned group-mixture pretraining corpus
FACED native32 downstream
frozen source-only probe
H = 200 / 500 / 1000 / 2000 h, seeds 0 / 1
```

- H200 remains at the random frozen-transfer floor.
- H500, H1000, and immutable H2000 are above random under paired target-subject uncertainty.
- Subject L1 is near ceiling throughout the pretrained range.
- All eight through-2000 cells pass the FACED task gate.
- No task-gated L5 subject intervention exceeds the strictly removed-variance-matched null after Holm correction.
- The global budget slope is positive but leave-one-budget-out signs are unstable; no monotonic scaling law.
- Historical mutable D2-2 H2000 values remain invalid; jobs 892861/892882 supersede them with SHA-pinned results.

## Questions

1. Does immutable H2000 sustain above-floor FACED transfer, create measurable subject reliance, or fall back?
2. Is the budget response caused by unique data volume, optimizer updates, or both?
3. Is the observed threshold specific to frozen linear accessibility or also present under full fine-tuning?
4. Does subject-first/task-later emergence replicate outside FACED, preferably in sleep staging?
5. Does the pattern generalize from CBraMod to a bounded CodeBrain replication?

## Ordered phases

### Phase A - H2000 immutable closure (COMPLETE)

Jobs 890151_6 and 890151_7 completed naturally. The closure gate pinned epochs 48/49 as read-only SHA-named
payloads, and jobs 892861/892882 reran and exactly reproduced the full FACED task/L1/L4/L5/L6 and paired
target-subject inference. H2000 sustains the floor crossing without a detected L5 effect beyond the matched null.
H4000 stays held.

### Phase B - Representation-emergence decomposition (NEXT)

Use existing valid checkpoints and source-only feature/probe fits. Pre-freeze continuous, non-ceiling endpoints:
subject log loss, pairwise AUC/margin, retrieval mAP, cross-subject class margin, class-probe log loss, variance
partition (subject/class/interaction/residual), and principal angles between subject- and task-predictive subspaces.
Include random and released references. Target test results may not select endpoints. This phase is exploratory.

### Phase C - Unique data x optimizer updates

First verify from logs whether equal epochs produced budget-dependent update counts. If so, freeze a 2x2 design:
200h-low, 200h-high, 1000h-low, 1000h-high. The two existing low/high corners are reused only if their update and
scheduler contracts match. LR schedules are defined by optimizer step. Seed 0 is a predeclared screen; seed 1 runs
only under a frozen screening gate. This phase is held until A and B review.

### Phase D - Frozen accessibility x fine-tuning adaptability

On the fixed FACED 1-80 / 81-100 / 101-123 split, compare random, H200, H500, H1000, immutable H2000, and released
under one downstream hyperparameter contract, source-val-only checkpoint selection, and fixed downstream seeds.
No per-budget sweep. This phase distinguishes linear accessibility from supervised adaptability. It is held until
Phase C's design is frozen and approved.

### Phase E - One cross-task validation

Use a gate-first sleep-staging candidate, with ISRUC-S3 as the literature-aligned first feasibility target. Gate on
random, released, H200, H1000, and immutable H2000. Require released > random, a non-floor task endpoint, a fixed
split/channel/sequence path, and adequate subject-cluster power. A failed released gate stops the task; no silent
third-dataset search. Held until A-D review.

### Phase F - Bounded CodeBrain replication (ACTIVE BY PM REPRIORITIZATION)

After the CBraMod mechanism questions are resolved, choose two or three budgets around the empirically determined
threshold. Test subject fingerprint emergence, task transfer, and task-gated L5. Do not make temporal-token claims
without resolving temporal-code collapse. Do not compare CodeBrain's 19-channel substrate unconditionally with
CBraMod Route B 33-channel results. The PM invoked the explicit reprioritization after B1-Core: the bounded
Stage-2 test is now the active candidate phase, while full 1k-9k replication remains a separate deferred project.
The frozen implementation and claim contract are S2P_23/S2P_24. Training remains held until their fail-closed
preflight passes and returns for PM review.

## Unified discipline

- Audit only immutable, SHA256-pinned checkpoints.
- Every downstream includes random-init and released references.
- Interpret L4/L5/L6 only after the frozen source-val task gate.
- Match removed variance exactly and compare subject intervention with a random-subspace null under a preregistered
  multiple-comparison correction.
- Target test labels are final scoring only, never selection.
- Budget comparisons use seed means and paired cluster inference, never best-seed claims.
- Do not substitute a new dataset after a downstream gate failure.

## Project-close gate

No project-close recommendation is allowed until all are resolved:

1. immutable H2000 closure and re-audit (complete);
2. data-volume versus optimizer-step decomposition;
3. frozen accessibility versus fine-tuning adaptability;
4. one gate-passed cross-task replication or a principled cross-task failure;
5. stable task-gated L5 over the final budget range;
6. bounded CodeBrain replication, or an explicit PM scope lock to CBraMod-only.

Even after these conditions, Codex must submit a distinct PROJECT-CLOSE recommendation before any writing phase.

## Prohibited work

Until explicit PM authorization after project-close review:

```text
no manuscript rewrite or skeleton
no abstract
no paper figures or submission tables
no venue formatting
no H4000
no new training outside an approved phase protocol
```
