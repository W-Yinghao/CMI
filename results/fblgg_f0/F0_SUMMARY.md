# CIGL_47 F0 — FBLGGGraph ERM-only gate (seeds 0/1/2)

**Verdict: BNCI2014_001 FAILS the ERM gate; BNCI2015_001 PASSES. F1 stays FROZEN.** Per the PI rule
("2a does not improve but 2015 improves → do NOT run F1; analyze the 4-class backbone bottleneck"), the
next step is a non-GPU BNCI2014 bottleneck analysis (CIGL_48), not any CMI method.

F0 tests only whether the **FBLGGGraph ERM backbone** is worth continuing as the SOTA-track backbone — it
is NOT a CMI experiment. Static `DGCNNGraph` remains closed as a diagnostic/audit baseline (the CIGL v0.6
bounded graph/node leakage-audit result); FB-LGG is the new SOTA-track backbone.

## Provenance

| field | value |
|---|---|
| Branch / SHA | `project/fb-lgg-dualcmi-scaffold` @ `1d9d519` (tracked tree clean at launch) |
| Environment | conda `eeg2025`; torch `2.6.0+cu124` (CUDA available on all jobs) |
| SLURM | `--partition=A100,V100,V100-32GB,A40 --gres=gpu:1`; private HOME per job; MNE readable mirror |
| Backbone / config | `FBLGGGraph` `erm:0` **only** (no graphcmi / graphdualpc / dec_scale / λ,γ / CDANN) |
| Datasets / folds / seeds | BNCI2014_001 `{0,1}`, BNCI2015_001 `{0,9}`, seeds `{0,1,2}` (12 cells) |
| Grouping | `central_strip_v1` on all cells (name-aware; no blob / no index fallback / no warning) |
| Job IDs | seed0 (dataset-level) `879143`,`879144`; seeds1/2 (target-level) `879153`–`879160` |

## Metric definitions (clarification requested by the PI)

- **`restored_source_bacc`** (reported as `source_bacc` in the raw JSON): balanced accuracy of the
  **restored best-epoch model** on the held-out source-probe split. This is the metric that reflects the
  model actually returned/used for target eval.
- **`final_train_source_bacc`**: the **last-epoch (ep 300)** training-set balanced accuracy — pre-restore.
  A high value here with low val/target is the source-memorization signature.
- **`best_source_val_bacc` / `final_val_source_bacc`**: source-VAL (held-out source subject) balanced
  accuracy at the best epoch / final epoch — the cross-subject transfer signal used for early stopping.

## Aggregate (12 cells; authoritative CSV: `F0_AGGREGATE.csv`)

| dataset | mean bAcc | worst | restored src bAcc | best src-val | final val | zero_graph | zero_temporal | permute_nodes | grouping | Δ vs DGCNN ERM |
|---|---|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 (.25) | **0.296** | 0.260 | 0.576 | 0.326 | 0.278 | 0.273 | 0.284 | 0.275 | central_strip_v1 | **−0.046** ✗ |
| BNCI2015_001 (.50) | **0.627** | 0.568 | 0.770 | 0.638 | 0.581 | **0.542** | 0.624 | **0.513** | central_strip_v1 | **+0.052** ✓ |

## Seed-level (authoritative CSV: `F0_SEED_TABLE.csv`)

| dataset | seed | tidx | subj | bAcc | best_ep | best_src_val | final_val | final_train |
|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 | 0 | 0 | 1 | 0.306 | 6 | 0.318 | 0.240 | 0.987 |
| BNCI2014_001 | 0 | 1 | 2 | 0.281 | 1 | 0.306 | 0.278 | 0.991 |
| BNCI2014_001 | 1 | 0 | 1 | 0.302 | 62 | 0.309 | 0.245 | 0.988 |
| BNCI2014_001 | 1 | 1 | 2 | 0.260 | 155 | 0.309 | 0.293 | 0.988 |
| BNCI2014_001 | 2 | 0 | 1 | 0.332 | 36 | 0.372 | 0.312 | 0.994 |
| BNCI2014_001 | 2 | 1 | 2 | 0.293 | 22 | 0.340 | 0.302 | 0.990 |
| BNCI2015_001 | 0 | 0 | 1 | 0.715 | 19 | 0.672 | 0.630 | 1.000 |
| BNCI2015_001 | 0 | 9 | 10 | 0.568 | 21 | 0.720 | 0.657 | 1.000 |
| BNCI2015_001 | 1 | 0 | 1 | 0.662 | 10 | 0.665 | 0.575 | 1.000 |
| BNCI2015_001 | 1 | 9 | 10 | 0.577 | 34 | 0.657 | 0.603 | 1.000 |
| BNCI2015_001 | 2 | 0 | 1 | 0.645 | 19 | 0.570 | 0.515 | 1.000 |
| BNCI2015_001 | 2 | 9 | 10 | 0.593 | 27 | 0.545 | 0.508 | 1.000 |

## Operational checks (all pass)

```
exit codes:        10/10 rc=0
CUDA:              cuda_avail True on all jobs (no CPU fallback)
mirror:            cigl_bnci_readable used on all jobs (no unreadable-datalake fallback)
braindecode/EEGNet/Shallow refs: 0
grouping_scheme:   central_strip_v1 (all 12 cells)      grouping_warning: 0
NaN/inf:           0        missing required metrics: 0
walltime:          seed0 ~9 min ; seeds1/2 ~5 min each (parallel)
JSON:              results/fblgg_f0/*.json (10 files, incrementally written)
```

## Gate mapping → verdict

- **BNCI2014_001: FAIL.** 0.296 < DGCNN 0.342 and near chance (0.25); not ≥ 0.40. Every cell fits
  source-train ~0.99 but **no epoch generalizes** — best-source-val is stuck near chance (0.31–0.37)
  across best_ep spanning 1→155, and target ≈ chance. Ablations (zero_graph/zero_temporal/permute_nodes
  all ~0.27–0.28) are near chance and near each other → neither branch carries transferable 4-class
  signal. This is a backbone/representation transfer problem, **not** grouping (now correct) or
  early-stopping (working) or CMI.
- **BNCI2015_001: PASS.** 0.627 vs 0.575 (**+5.2pp**); the **graph branch is load-bearing**
  (zero_graph 0.627→0.542, permute_nodes→0.513; zero_temporal barely moves it). A genuine graph-driven
  improvement over static DGCNN on the binary task.

## Decision / next action

**F1 frozen** (no graphcmi / graphdualpc / dec_scale=300 / λ,γ sweep / more seeds / extra folds /
EEGNet-Shallow / DGCNN reruns / dynamic edge). Next: **CIGL_48 non-GPU BNCI2014 4-class bottleneck
analysis** (branch `project/fblgg-2a-bottleneck-analysis`) — data/split sanity, class-wise confusion,
branch/fusion ablation deltas, and a CPU CSP/LDA classical sanity baseline to localize the bottleneck
(temporal stem / graph branch / fusion / data-fold difficulty / 4-class label structure) before any
architecture change. `dec_scale=300` remains the F1 default *candidate* only, unused until a backbone
clears the 4-class gate.
