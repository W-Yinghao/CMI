# CIGL Phase 3A-H — DGCNN Adapter Graph/Node Leakage Audit Results (BNCI2014_001, fold-0)

> **EXPLORATORY diagnostic — NOT a benchmark / SOTA result, and NOT a regularizer.** One dataset, one
> LOSO fold, 3 seeds, source-only, ERM only. Asks whether the **task-capable** static DGCNN adapter's
> learned `graph_z`/`node_z` carry label-conditional source-domain leakage `I(Z;D|Y)`.

## 1–6. Run provenance

```bash
sbatch scripts/sbatch_cigl_phase3a_dgcnn_leakage_audit_bnci001.sh
# -> python scripts/run_cigl_phase3a_dgcnn_leakage_audit.py --dataset BNCI2014_001 --device cuda --fold 0 \
#      --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05
```

| field | value |
|---|---|
| SLURM job id | **876784** |
| partition / node | multi-partition `A100,V100,V100-32GB,A40` (default QOS) → scheduled on **node07** |
| runtime | ~30–40 min (sacct unavailable; log mtime 12:24; n_perm=50 graph+node audit over 3 seeds) |
| branch / commit_hash | `project/cigl-phase3a-dgcnn-leakage-audit` / `a75edc14430ab0db9211dd9f107a974ed789906d` |
| config_hash | `5a1210e8e68d` |
| environment | conda `eeg2025`, torch 2.6.0+cu124 (CUDA) |
| dataset / fold / held-out target | BNCI2014_001 / fold-0 / **subject 1** (never used in training/extraction/probe) |
| candidate | `dgcnn_forward_graph_adapter` (static/shared adjacency) |
| seeds / classes / chance / floor | 0,1,2 / 4 / **0.25** / floor **0.45** · n_perm=**50** · gate_alpha=**0.05** |

## 7. DGCNN task table (ERM; per-seed; means over 3 seeds)

| seed | train bAcc | source_probe bAcc/F1 | target_eval bAcc/F1 (eval-only) | train−src gap |
|---|---|---|---|---|
| 0 | 0.757 | 0.481 / 0.481 | 0.436 / 0.436 | +0.276 |
| 1 | 0.741 | 0.451 / 0.451 | 0.438 / 0.424 | +0.290 |
| 2 | 0.737 | 0.442 / 0.441 | 0.420 / 0.415 | +0.295 |
| **mean** | **0.745** | **0.458** | **0.431** | **+0.287** |

`task_ok = true` (mean source 0.458 ≥ 0.45; **2/3** seeds individually ≥ 0.45 — seed2 0.442 just below).
This **reproduces the Phase 3A-G DGCNN result (0.458)** — a credible but **modest** task baseline.

## 8. Graph-usage checks (both deltas; source probe-pool)

| seed | zero_graph Δ | permute_nodes Δ |
|---|---|---|
| 0 | +0.231 | +0.241 |
| 1 | +0.201 | +0.203 |
| 2 | +0.192 | +0.190 |
| **mean** | **+0.208** | **+0.211** |

`graph_path_used = true` — zeroing the graph readout (and permuting node content) collapses source toward
chance, so logits route through `graph_z`; **no CNN bypass**.

## 9. Graph leakage `I(Z_g;D|Y)` proxy (per seed; `clears_null = kl_mean > permutation_mean AND p ≤ 0.05`)

| seed | kl_mean | permutation_mean | permutation_p | clears_null |
|---|---|---|---|---|
| 0 | 1.271 | 0.174 | 0.020 | **True** |
| 1 | 1.265 | 0.160 | 0.020 | **True** |
| 2 | 1.245 | 0.144 | 0.020 | **True** |

**Graph leakage clears the null in 3/3 seeds** — observed KL (~1.26) ≈ **8×** the permutation mean
(~0.16); `p=0.020` is the n_perm=50 resolution floor (observed exceeded **all 50** within-label permuted
refits). `graph_leakage_exists = true`.

## 10. Node leakage `(1/C) Σ_v I(Z_v;D|Y)` proxy (per seed)

| seed | kl_mean | permutation_mean | permutation_p | clears_null |
|---|---|---|---|---|
| 0 | 0.518 | 0.035 | 0.020 | **True** |
| 1 | 0.502 | 0.033 | 0.020 | **True** |
| 2 | 0.544 | 0.033 | 0.020 | **True** |

**Node leakage clears the null in 3/3 seeds** (KL ~0.52 ≈ **15×** the permutation mean ~0.034; `p=0.020`).
`node_leakage_signal = true`.

## 11. Node-map stability

| mean_corr | null_q95 | above_random | degenerate | node_leakage_claimed |
|---|---|---|---|---|
| **0.945** | 0.202 | **True** | **False** | **True** |

The per-channel `node_leakage_map` is **highly reproducible across seeds** (mean pairwise correlation
0.945 ≫ channel-permutation null q95 0.202) and **non-degenerate** → the node leakage is **spatially
localized and stable**, so a node-level claim is supported (not just a per-seed signal).

## 12. Edge — explicitly skipped (not faked)

`edge_logits_dynamic = false`, `edge_audit_skipped = true`,
`edge_skip_reason = "static/shared adjacency: edge_logits=None; no per-sample edge object"`. No
dynamic-edge leakage is computed or claimed.

## 13. Firewall flags

`used_target_labels_for_training=false`, `used_target_labels_for_selection=false`,
`used_target_covariates=false`, `target_eval_is_evaluation_only=true`, `cmi_regularization_used=false`.
(`audit_passes=true`.)

## 14. Recommended decision — **A** *(pending reviewer)*

**Decision A: the task-capable DGCNN adapter passes and carries strong graph *and* node leakage → a
graph/node regularizer pilot may be considered (no edge-CMI — the edge object is absent).** This is the
**first** point in the project where the two prerequisites for a meaningful regularizer pilot hold at
once: (i) a graph backbone that actually learns the task (source 0.458, ≥2/3 seeds, graph path used, no
bypass), and (ii) **verified, significant, stable** label-conditional domain leakage on that backbone —
graph KL ~8× / node KL ~15× the permutation mean, `p=0.020` in 3/3 seeds, with a node map stable at
mean_corr 0.945. All prior leakage evidence sat on **near-chance or overfit** encoders (GraphCMINet, the
dynamic stems); this is the first on a task-capable one.

**Honest caveats (do not overclaim):**
- The task baseline is **modest** — 0.458 mean, and seed2 (0.442) dips just below the 0.45 floor; the
  known-good CNNs reached 0.52–0.56. So the task-vs-leakage **tradeoff headroom is limited**; a pilot
  must watch task cost carefully and not declare a method win on a thin baseline.
- This is **graph/node only**. The DGCNN adjacency is static, so **edge-CMI remains out of scope**.
- Phase 3A-H establishes graph/node leakage as a **valid regularization target** on a task-capable
  graph backbone — but controllability is **not** proven here. Whether graph/node CMI regularization can
  reduce this leakage without damaging the modest task baseline remains the next pilot question (only a
  regularizer pilot can establish controllability).

**If authorized (reviewer-gated), the next step would be a graph/node CMI regularizer pilot on the DGCNN
adapter** (small λ ladder, source-only selection, the same firewall) — **not** edge-CMI, **not** full
LOSO, **not** SEED, **not** a λ-grid/SOTA. The CIGL regularizer remains **not authorized** until you
review this result. Generated per-seed JSON are gitignored; this doc is the tracked record.
