# CIGL_33 — Evidence Index

> Phase 4A consolidation (docs only). Maps each gate/phase to its branch, tracked doc, setup, decision,
> and exactly what it does / does not support. Authority for line counts/bytes is `git show origin` /
> the GitHub blob page (not the branch-ref raw preview).

All phases: strict source-only DG; target labels evaluation-only; conda `eeg2025`; multi-partition
`A100,V100,V100-32GB,A40`, default QOS. Leakage = posterior-KL proxy vs a retrained within-label
permutation null; `clears_null = kl_mean > permutation_mean AND permutation_p ≤ 0.05`.

## Per-phase index

| phase | branch | tracked doc | dataset | configs | seeds | n_perm | decision |
|---|---|---|---|---|---|---|---|
| Gate-2 | cigl-gate2-evidence | CIGL (gate-2 docs) | BNCI2014_001 fold-0 | GraphCMINet-ERM audit | 0,1,2 | 50 | leakage EXISTS (graph/node/edge) |
| 3A | cigl-phase3a-pilot | CIGL_15 / CIGL_16 | BNCI2014_001 fold-0 | graph/node/edge λ ladder | 0,1,2 | 20/50 | controllability PASS, **task FAIL** |
| 3A-R | cigl-phase3a-baseline-repair | CIGL_18 | BNCI2014_001 fold-0 | 6 GraphCMINet-ERM repairs | 0,1,2 | 20/50 | baseline repair **FAIL** (≈chance) |
| 3A-S | cigl-phase3a-backbone-sanity | CIGL_21 | BNCI2014_001 fold-0 | 5 known-good decoders (ERM) | 0,1,2 | 10 (graph ref) | protocol learnable; GraphCMINet is the bottleneck |
| 3A-G | cigl-phase3a-graph-backbone-redesign | CIGL_23 | BNCI2014_001 fold-0 | 3 graph-compat backbones (ERM) | 0,1,2 | 10 | **only static DGCNN adapter** task-capable |
| 3A-H | cigl-phase3a-dgcnn-leakage-audit | CIGL_25 | BNCI2014_001 fold-0 | DGCNN adapter audit | 0,1,2 | 50 | DGCNN graph/node leakage **EXISTS** |
| 3A-I | cigl-phase3a-dgcnn-gn-regularizer-pilot | CIGL_27 | BNCI2014_001 fold-0 | 8 graph/node λ (pilot) | 0,1,2 | 20/50 | `graph_node_010` pilot **PASS** |
| 3A-J | cigl-phase3a-dgcnn-gn-multifold-confirmation | CIGL_29 | BNCI2014_001 folds 0–8 | fixed `graph_node_010` | 0,1,2 | 50 | **CONFIRMED** (primary folds 1–8) |
| 3A-K | cigl-phase3a-dgcnn-gn-second-dataset-confirmation | CIGL_31 | BNCI2015_001 (12 folds) | fixed `graph_node_010` | 0,1,2 | 50 | **CONFIRMED** (`confirmed_with_target_guardrail`) |

## What each phase supports / does NOT support

- **Gate-2 — supports:** graph/node/edge objects in GraphCMINet carry label-conditional domain leakage.
  **Not:** that GraphCMINet is a usable task model (it is near-chance).
- **3A — supports:** the leakage is *controllable* (graph/node ~99.9% reducible). **Not:** a
  task-preserving method on GraphCMINet (task collapses 3.7–7.2 pt; ERM ≈ chance 0.33).
- **3A-R — supports:** GraphCMINet under-fits 4-class MI; 6 training-only repairs do not fix it (all
  src ≈ 0.33 < 0.45 floor; controls pass → underfitting, not a probe/protocol artifact). **Not:** that the
  protocol/data are unlearnable.
- **3A-S — supports:** the *protocol is learnable* (EEGNet/ShallowConvNet/DeepConvNet/DGCNN all clear
  source ≥ 0.45 while GraphCMINet stays 0.334) → GraphCMINet is the bottleneck. **Not:** that any graph
  model with dynamic edges works.
- **3A-G — supports:** among graph-compatible backbones, **only** the static DGCNN adapter learns the task
  (≥0.45, graph path used, no bypass); dynamic-edge stems **overfit** (train ≈ 1.0, source ≈ chance).
  **Not:** a dynamic-edge / edge-CMI path (out of scope thereafter).
- **3A-H — supports:** on the task-capable DGCNN, `graph_z`/`node_z` carry **strong, significant, stable**
  leakage (graph KL ≈ 8× / node ≈ 15× perm; p at n_perm=50 floor; node-map corr 0.945). **Not:** edge
  leakage (static adjacency; audit skipped). **Not:** controllability (that needs a regularizer).
- **3A-I — supports:** `graph_node_010` reduces graph 48% / node 42% (3/3 seeds, confirmed n_perm=50) with
  source retained on **fold-0** — first task-preserving reduction. **Not:** generality (single dev fold).
- **3A-J — supports:** the fixed candidate **replicates across BNCI2014_001** (folds 1–8 primary 8/8;
  fold-0 = dev, excluded). **Not:** cross-dataset (one dataset).
- **3A-K — supports:** the fixed candidate **replicates on a second dataset (BNCI2015_001)** with the
  target guardrail held (12/12), `confirmed_with_target_guardrail=true`. **Not:** SOTA, edge-CMI,
  cross-architecture, or leakage elimination (reduction is partial; null still clears).

## Cross-cutting caveats (apply to the whole chain)

- Partial controllability, not erasure — regularized leakage still clears the null every confirmation fold.
- Posterior-KL **proxy**, not unbiased CMI.
- Two **MI** datasets; one fixed config; graph/node only; DGCNN static-adjacency backbone; modest baselines.
