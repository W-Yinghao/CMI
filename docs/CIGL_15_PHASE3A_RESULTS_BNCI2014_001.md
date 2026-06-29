# CIGL Phase 3A — Exploratory Regularizer-Effect Results (BNCI2014_001, fold-0)

> **EXPLORATORY pilot evidence — NOT a benchmark / accuracy / SOTA result.** One dataset, one LOSO
> fold, a fixed 7-config set, 3 seeds. Source-only. No λ grid. It answers the *controllability*
> question (can CMI regularization reduce the Gate-2 leakage without destroying task performance?) and
> informs the next-step decision; it is **not** a method claim.

## Run provenance

```bash
sbatch scripts/sbatch_cigl_phase3a_bnci001.sh
# -> python scripts/run_cigl_phase3a_regularizer_pilot.py --dataset BNCI2014_001 --device cuda \
#      --fold 0 --seeds 0 1 2 --n_perm 20 --n_perm_confirm 50 --epochs 80 --probe_epochs 100 --gate_alpha 0.05
```

| field | value |
|---|---|
| SLURM job id | **876274** |
| partition / node | A100 / node03 (A100-PCIE-40GB) — multi-partition submit, default QOS |
| runtime | ~33 min |
| branch / commit_hash | `project/cigl-phase3a-regularizer-pilot` / `3602f9442ceab2cf15d0309fc9d08ebbbfaa5814` |
| config_hash | `8019a4ef2859` |
| environment | conda `eeg2025`, torch 2.6.0+cu124 (CUDA) |
| dataset / fold / held-out target | BNCI2014_001 / fold-0 / **subject 1** (never used in training / extraction / probe) |
| source subjects (domains) | 2–9 (**8 domains**) |
| seeds | 0, 1, 2 |
| pass-1 / confirmation n_perm | 20 / 50 |
| epochs / probe_epochs / gate_alpha | 80 / 100 / 0.05 |
| three-way source split (per seed) | enc-train 3232 → frozen; probe-pool 1376 → support-aware (Y,D) split 960/416; 8/8 domains in both, 0 low-support |

Strict source-only confirmed: `used_target_labels_for_training=false`,
`used_target_labels_for_selection=false`, `used_target_covariates=false`,
`target_eval_is_evaluation_only=true`. (The target-label rule is additionally **behaviorally** tested —
corrupting target labels leaves every source-side number byte-identical.)

## How to read the numbers

- **`reduction%`** = `(ERM_kl − config_kl) / ERM_kl` per object (the regularizer-effect metric).
- **`clears_null`** (count over 3 seeds) is the Gate-2 *existence* gate. ⚠️ At `kl ≈ 0` it can still read
  `3/3` because the tiny residual exceeds an even-tinier permutation null — so **`clears_null` is NOT
  the right metric for "leakage was removed"**; the **magnitude (`kl_mean`) and `reduction%`** are.
- `src bAcc` = held-out **source** balanced accuracy; `tgt bAcc` = **target** balanced accuracy
  (**evaluation-only**, never used for training/selection). 4-class chance = 0.25.

## Pass-1 results (all 7 configs, n_perm=20; reductions vs pass-1 ERM)

| config | src bAcc | tgt bAcc | graph kl (red%) | node kl (red%) | edge kl (red%) | clears g/n/e |
|---|---|---|---|---|---|---|
| **erm** (ref) | 0.329 | 0.324 | 0.553 (0%) | 0.545 (0%) | 0.878 (0%) | 3/3/3 |
| graph_only | 0.270 | 0.284 | 0.0003 (**100%**) | 0.0006 (**99.9%**) | 0.110 (87.5%) | 3/3/3 |
| node_only | 0.275 | 0.274 | 0.0004 (99.9%) | 0.0009 (99.8%) | 0.0003 (100%) | 3/3/1 |
| **edge_only** | **0.330** | **0.337** | **0.451 (18.6%)** | **0.451 (17.2%)** | 0.013 (98.6%) | 3/3/3 |
| graph_node | 0.257 | 0.259 | 0.0001 (100%) | 0.0001 (100%) | 0.0012 (99.9%) | 3/3/3 |
| full_cigl | 0.266 | 0.259 | 0.0000 (100%) | 0.0000 (100%) | 0.0018 (99.8%) | 3/3/2 |
| low_full_cigl | 0.292 | 0.273 | 0.0005 (99.9%) | 0.0017 (99.7%) | 0.0000 (100%) | 3/3/1 |

## Confirmation (n_perm=50; reductions vs **confirmation** ERM) — ERM, full_cigl, best-Pareto

`best_pareto_config = low_full_cigl` (source-only selection: largest graph+node reduction penalized by
source-task drop). Per-seed confirmation records are retained
(`confirmation_per_seed`, `*_confirm_*_seed*_nperm50.json`).

| config | src bAcc | tgt bAcc | graph kl (red%) | node kl (red%) | edge kl (red%) | clears g/n/e |
|---|---|---|---|---|---|---|
| erm | 0.328 | 0.324 | 0.553 (0%) | 0.545 (0%) | 0.880 (0%) | 3/3/3 |
| full_cigl | 0.260 | 0.253 | 0.0000 (100%) | 0.0000 (100%) | 0.0023 (99.7%) | 2/3/2 |
| low_full_cigl | 0.292 | 0.296 | 0.0007 (99.9%) | 0.0025 (99.5%) | 0.029 (96.7%) | 3/3/3 |

Confirmation observed `kl_mean` matches pass-1 to ~`1e-4` (the determinism fix; residual is GPU cudnn
non-determinism, exact on CPU) — i.e. this is the same frozen model re-audited at higher permutation
power, not a different network.

## Findings

1. **Leakage is strongly CONTROLLABLE.** Every config with a graph or node term drives graph **and**
   node leakage from ~0.55 down to ~`1e-3`–`1e-4` (**≈99.9–100% reduction**), and usually edge too.
   The graph/node CMI regularizers do exactly what they are designed to do.
2. **But the reduction costs source-task accuracy.** All graph/node configs drop source bAcc by
   **3.7–7.2 points** (graph_node −7.2, full_cigl −6.3, graph_only −5.9, node_only −5.4,
   low_full_cigl −3.7). Even the gentlest (λ=0.1) exceeds the **≤3-pt** pass threshold.
3. **`edge_only` is the only task-preserving config** (src 0.330 ≈ ERM 0.329, tgt +1.3): it removes
   edge leakage (98.6%) but leaves **graph/node leakage ~80% intact** (only 17–19% reduction). So the
   edge term alone does not control the dominant (graph/node) leakage.
4. **⚠️ The task baseline is near chance.** GraphCMINet-ERM source bAcc is **0.329**, barely above
   4-class chance (0.25). The leakage reductions are real and large, but the *task-cost* is measured in
   a near-chance regime, so the absolute task numbers (0.26–0.33) and the task-vs-leakage Pareto are
   **fragile and not yet trustworthy as a method verdict**.

## Pass / fail (per `docs/CIGL_14` §criteria) — Claude's reading, pending reviewer

| criterion | verdict |
|---|---|
| 1. ≥30% graph **or** node reduction in ≥2/3 seeds | **PASS** (≈100% for all graph/node configs) |
| 2. source bAcc drop ≤3 pt | **FAIL** (best reducer low_full_cigl is −3.7; edge_only passes but barely reduces graph/node) |
| 3. target bAcc drop ≤5 pt | mixed/**FAIL** (graph_node −6.5; low_full_cigl pass-1 −5.1 / confirm −2.8 — noisy near chance) |
| 4. fresh held-out audit probes (not Step-A) | **PASS** |
| 5. graph_node or full_cigl Pareto-improving | **FAIL** (they control leakage but cost task) |

**No config satisfies criteria 1 AND 2 together** → the strict Phase-3A gate does **not** pass.

## Recommended next path — **NOT Path A; re-pilot on a fixed baseline** *(pending reviewer decision)*

Controllability is proven, but the favorable-tradeoff gate fails *and* the GraphCMINet-ERM baseline is
near chance, so the current evidence cannot support a full-CIGL method claim (Path A) or a clean
graph+node/node-only narrowing (B/C) — the task-cost numbers are not yet interpretable.

Recommended before any Phase-3-proper decision:

1. **Establish a non-degenerate task baseline first.** GraphCMINet-ERM at ~0.33 (≈chance) is the
   confound. Strengthen the encoder/training (more epochs, backbone/optimizer tuning, possibly a
   stronger graph backbone) so source bAcc is clearly above chance; only then is the task-vs-leakage
   tradeoff meaningful.
2. **Re-pilot with a gentler λ sweep** (e.g. λ ≤ 0.05) once the baseline holds — the λ=0.1–0.3 tested
   here are too strong (they crush leakage AND task).
3. **Watch `edge_only`** as the task-preserving narrow direction: it removes edge leakage at no task
   cost, so an Edge-CMI story may survive a stronger baseline — but on its own it does not address the
   dominant graph/node leakage.

This keeps the claim narrower than the evidence: **leakage is controllable; favorable task-preserving
control is not yet demonstrated on a trustworthy baseline.** No Phase-3 / full-LOSO / SEED / λ-grid work
is warranted until the baseline is fixed and re-piloted. Generated per-config/seed JSON and `.npy` are
gitignored; this doc is the tracked review record.
