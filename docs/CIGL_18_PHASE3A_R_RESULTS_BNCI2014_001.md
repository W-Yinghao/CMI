# CIGL Phase 3A-R — Exploratory Baseline-Repair Results (BNCI2014_001, fold-0)

> **EXPLORATORY pilot evidence — NOT a benchmark / SOTA result.** One dataset, one LOSO fold, 3 seeds,
> source-only. Part A asks whether a non-degenerate GraphCMINet-ERM baseline exists; Part B (gentle
> re-pilot) runs only if Part A passes. This informs the next-step decision; it is not a method claim.

## Run provenance

```bash
sbatch scripts/sbatch_cigl_phase3a_baseline_repair_bnci001.sh
# -> python scripts/run_cigl_phase3a_baseline_repair.py --dataset BNCI2014_001 --device cuda --fold 0 \
#      --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 20 --n_perm_confirm 50
```

| field | value |
|---|---|
| SLURM job id | **876328** |
| partition / node | scheduled on **node26** (multi-partition `A100,V100,V100-32GB,A40`, default QOS) |
| runtime | ~15–20 min (Part A only; Part B skipped) |
| branch / commit_hash | `project/cigl-phase3a-baseline-repair` / `7f561b27cdc22b9521e1523cebfcf0d44c1e32da` |
| config_hash | `192d7de0a48e` |
| environment | conda `eeg2025`, torch 2.6.0+cu124 (CUDA) |
| dataset / fold / held-out target | BNCI2014_001 / fold-0 / **subject 1** (never used in training/extraction/probe) |
| source subjects | 2–9 (8 domains); enc-train 3232, probe-pool 1376 |
| seeds / classes / chance | 0,1,2 / 4 / **0.25** |

Strict source-only flags: `used_target_labels_for_training=false`,
`used_target_labels_for_selection=false`, `used_target_covariates=false`,
`target_eval_is_evaluation_only=true`.

## Part A — baseline candidate table (GraphCMINet-ERM; per-config means over 3 seeds)

(`bAcc` shown; macro-F1 ≈ bAcc throughout, all near chance. `tgt` = target_eval, **evaluation-only**.)

| candidate | train bAcc | src_probe bAcc | tgt bAcc | train−src gap | graph KL | node KL | edge KL |
|---|---|---|---|---|---|---|---|
| current_default | 0.391 | 0.334 | 0.328 | +0.057 | 0.529 | 0.503 | 0.894 |
| source_channel_zscore | 0.391 | 0.334 | 0.328 | +0.057 | 0.529 | 0.503 | 0.895 |
| stronger_graphcmi_backbone | 0.409 | 0.328 | 0.317 | +0.081 | 0.734 | 0.725 | 0.795 |
| lower_lr_longer | 0.378 | 0.329 | 0.327 | +0.048 | 0.535 | 0.529 | 0.854 |
| no_classbal_sampler | 0.397 | 0.327 | 0.313 | +0.070 | 0.584 | 0.570 | 0.874 |
| ce_balance_check | 0.391 | 0.334 | 0.328 | +0.058 | 0.529 | 0.502 | 0.894 |

## Controls

| control | value | interpretation |
|---|---|---|
| overfit_small_source train bAcc | **0.531** | architecture **can** overfit a tiny balanced subset (≫ chance 0.25) |
| label_shuffle_control src bAcc | **0.256** | shuffled source labels → **≈ chance**; the probe is not cheating |
| controls_ok | **true** | both controls behave |

## Baseline gate

- `baseline_gate_pass = **false**` · `best_baseline = **None**` (floor 0.45). **No candidate** reaches
  source_probe bAcc ≥ 0.45 (or ≥ current_default + 0.10) — all sit at ~0.33, barely above 4-class chance.
- **Part B (gentle re-pilot) SKIPPED.** Stop reason: *baseline gate FAILED → recommend
  architecture/preprocessing diagnosis (do not claim CIGL as a method).*
- Because Part B did not run, there are no `source_only_reducers` / `confirmation_labels` /
  `final_task_preserving_reducers`; the selection firewall (`confirmation_label_selection_uses_target_eval
  = false`) is moot for this run.

## Diagnosis

The architecture is **not broken** (the overfit control reaches 0.53; label-shuffle gives chance), but
on the full source it **underfits**: train bAcc is only ~0.39 on a 4-class task and the train−source gap
is tiny (~0.05), so this is **underfitting, not overfitting**. **None** of the six repairs lift source
bAcc above ~0.33:

- `source_channel_zscore` is a **no-op** — `moabb_data` already trial-z-scores the input, so a
  per-channel z-score is ≈ identity (results byte-identical to `current_default`).
- a bigger net (`stronger_graphcmi_backbone`), a slower/longer schedule (`lower_lr_longer`, 150 epochs),
  a raw sampler, and balanced CE all leave source bAcc ≈ chance (and leakage still strong, graph KL
  0.53–0.73, edge ~0.85–0.89).

So the **task baseline is not repairable** with these candidates: GraphCMINet-ERM cannot learn
BNCI2014_001 4-class MI to a non-degenerate level under strict source-only training. (Leakage remains
large and controllable — Phase 3A — but a task-preserving method cannot be evaluated on a near-chance
backbone.)

## Recommended next path — **C → D** *(Claude's recommendation, pending reviewer)*

Per `docs/CIGL_17` decision branches:

- **A (full graph/node/full CIGL): NO.** **B (Edge-CMI narrow): NO.** Both require a credible baseline.
- **C — baseline diagnosis (immediate):** GraphCMINet-ERM underfits 4-class MI here. Diagnose/replace the
  task backbone before any method claim — e.g. benchmark a **known-good MI decoder** (EEGNet / DGCNN /
  ShallowConvNet) on the *same* strict source-only fold-0 protocol; if those reach the usual ~0.5–0.7
  source bAcc while GraphCMINet stays ~0.33, the problem is GraphCMINet's capacity/optimization as an MI
  decoder, not the protocol. (The candidate set tried here did not fix it.)
- **D — diagnostic framework (fallback):** if no source-only-trainable backbone yields a non-degenerate
  MI baseline, CIGL's defensible contribution is the **leakage audit / diagnostic** (Gate-2: learned EEG
  graph objects carry significant, controllable subject leakage), **not** a training method.

**Bottom line (claim kept narrower than evidence):** leakage exists and is controllable, but on this
setup CIGL cannot be shown to be a *task-preserving* method because the GraphCMINet-ERM baseline is
near-chance and not repairable by the tested candidates. No Phase-3 / full-LOSO / SEED / λ-grid is
warranted until a credible source-only task baseline is established (Path C) — otherwise reframe to the
diagnostic contribution (Path D). Generated per-candidate JSON / `.npy` are gitignored; this doc is the
tracked record.
