# CIGL_36 — Reproducibility Checklist

> Phase 4A consolidation (docs only). How to regenerate the two confirmation results and verify the
> source-only firewall. No new experiments here.

## Environment

- conda env `eeg2025` (torch 2.6.0+cu124). CPU dry-runs use `torch.set_num_threads(1)`.
- SLURM: multi-partition `A100,V100,V100-32GB,A40`, **default QOS, no `--time`**; fail-closed if no CUDA.
- Data: read-only datalake `/projects/EEG-foundation-model/datalake/raw`. `cmi/data/moabb_data.py`
  imports `cmi.paths.configure_offline_moabb()` (offline, no download).

## Key evidence (branch / tracked doc / job)

| phase | branch | doc | SLURM job | node |
|---|---|---|---|---|
| 3A-H audit | cigl-phase3a-dgcnn-leakage-audit | CIGL_25 | 876784 | node07 |
| 3A-I pilot | cigl-phase3a-dgcnn-gn-regularizer-pilot | CIGL_27 | 876887 | node09 |
| 3A-J confirm (2a) | cigl-phase3a-dgcnn-gn-multifold-confirmation | CIGL_29 | 876990 | node09 |
| 3A-K confirm (2015) | cigl-phase3a-dgcnn-gn-second-dataset-confirmation | CIGL_31 | 877369 | node12 |

(Use `git show origin/<branch>:<file> | wc -l` or the GitHub blob page as the byte/line authority — not
the branch-ref raw preview.)

## Fixed method config (frozen for all confirmation)

```
backbone   = dgcnn_forward_graph_adapter   (static/shared adjacency; edge_logits=None)
config     = graph_node_010                (λ_g=0.010, λ_node=0.010, λ_edge=0.000)
baseline   = erm_fixed                     (λ_g=λ_node=0)
seeds      = 0 1 2
epochs     = 80    probe_epochs = 100
n_perm     = 50    gate_alpha = 0.05
audit      = audit_graph_node_objects (graph+node only; within-label retrained permutation null)
```

## Datasets / preprocessing

- **BNCI2014_001** (4-class, chance 0.25): MotorImagery, resample 128, tmin/tmax 0.5/3.5, band 4–38,
  trial z-score. LOSO subjects 1–9; **fold-0 = dev** (selected `graph_node_010`), **primary = folds 1–8**.
- **BNCI2015_001** (binary right_hand/feet, chance 0.50): **MotorImagery(events=["right_hand","feet"],
  n_classes=2)** (LeftRightImagery is invalid — not left/right hand), resample 128 (matches 2a protocol;
  250 Hz is a known note, not used), tmin/tmax 0.5/3.5, band 8–30, trial z-score, interval [0,5] →
  window [0.5,3.5] inside. **12 LOSO folds, all confirmation (no dev fold).**
- **Datalake mirror note:** BNCI2015_001's lampx mirror (`~bci/database/001-2015/`) is owner-locked; the
  sbatch builds a **readable symlink mirror** (symlinks only; no download; no data copy) pointing at the
  datalake's readable bnci-horizon copy `database/data-sets/001-2015/`, then exports
  `MNE_DATASETS_BNCI_PATH`/`MNE_DATA`. Verified by `--preflight_only` (binary, 12 subj, 13 ch).

## Commands to regenerate

```bash
# Phase 3A-J — BNCI2014_001 multi-fold confirmation
sbatch scripts/sbatch_cigl_phase3a_dgcnn_gn_multifold_confirmation_bnci001.sh
#  -> --folds 0 1 2 3 4 5 6 7 8 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05

# Phase 3A-K — BNCI2015_001 second-dataset confirmation (preflight first)
python scripts/run_cigl_phase3a_dgcnn_gn_second_dataset_confirmation.py --dataset BNCI2015_001 --device cpu --preflight_only
sbatch scripts/sbatch_cigl_phase3a_dgcnn_gn_second_dataset_confirmation_bnci2015_001.sh
#  -> --dataset BNCI2015_001 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05
```

Do not reduce folds/seeds/epochs/probe_epochs/n_perm; do not change λ; do not switch dataset; if a real
dataset is not binary the runner stops before training (re-authorization required).

## Source-only firewall checklist (must all hold)

- [ ] `used_target_labels_for_training = false`
- [ ] `used_target_labels_for_selection = false`
- [ ] `used_target_covariates = false`
- [ ] `target_eval_is_evaluation_only = true`
- [ ] `selection_uses_target_eval = false`
- [ ] `confirmation_label_selection_uses_target_eval = false`
- [ ] configs are **fixed** (`erm_fixed`, `graph_node_010`); no λ-grid / no selection in confirmation
- [ ] target enters **only** the reported guardrail (`target_guardrail_pass`); Decision A requires
      `confirmed_with_target_guardrail = true`
- [ ] target-label-corruption tests pass (source metrics + source-only verdict invariant)
- [ ] edge: `edge_regularization_used=false`, `edge_audit_skipped=true`, no edge-CMI claim

## Decision rule (binding)

`source_only_confirmed = (ERM adequacy & ERM leakage & reg reduces & source retained)` over the primary
folds; `confirmed_with_target_guardrail = source_only_confirmed AND target_guardrail_pass`. **Decision A
only if `confirmed_with_target_guardrail = true`.** ERM-adequacy failure → D (preprocessing/backbone
diagnosis; no silent window/resample/dataset change).

## Generated-artifact policy

- Per-seed/summary JSON and `.npy` are **gitignored** (`results/cigl/**/*.json`); the tracked record is
  the `docs/CIGL_*` doc. Paper tables (`scripts/collect_cigl_evidence_tables.py` → `results/cigl/paper_
  tables/`) are also gitignored unless explicitly approved.
