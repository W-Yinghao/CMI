# `results/cigl/phase3a_baseline_repair/` — Phase 3A-R artifacts

Outputs of `scripts/run_cigl_phase3a_baseline_repair.py` (see
`docs/CIGL_17_PHASE3A_R_BASELINE_REPAIR.md`).

> **EXPLORATORY artifacts, NOT benchmark results.** Every JSON carries `meta.exploratory = true`,
> `meta.phase = "Phase3A_R_baseline_repair"`, `meta.setting = "strict_source_only_DG"`. Part A asks
> whether a non-degenerate GraphCMINet-ERM baseline exists; Part B (only if A passes) tests a gentle
> CMI micro-ladder. These inform the next-step decision; they are not a method claim.

The generated `*.json` here are **gitignored** (regenerable); only this README is tracked.

## Files

```
results/cigl/phase3a_baseline_repair/
  README.md                                              # (tracked) this file
  <dataset>_fold<F>_baseline_<candidate>_seed<S>.json    # Part A: ERM baseline candidate per seed
  <dataset>_fold<F>_gentle_<config>_seed<S>_nperm<N>.json# Part B: gentle micro-ladder per seed (if A passes)
  <dataset>_fold<F>_baseline_repair_summary.json         # Part A + (conditional) Part B
```

## Summary JSON

- `meta`: exploratory / phase / setting + strict-target-label flags + chance + commit/config hash.
- `part_a`:
  - `candidates[name]`: `train_bacc`, `source_probe_bacc`, `target_eval_bacc` (eval-only),
    `train_minus_source_gap`, `graph/node/edge_kl`.
  - `controls`: `overfit_small_source_train_bacc` (want ≫ chance), `label_shuffle_control_src_bacc`
    (want ≈ chance), `controls_ok`.
  - `baseline_bacc_floor`, `baseline_gate_pass`, `best_baseline` (source-only selection).
- `part_b` (**null if Part A fails**): `baseline`, `erm_reference_kl`, `gentle[config]`
  (`source_probe_bacc`, `source_drop_vs_erm`, `target_drop_vs_erm`, `{obj}_kl_mean`,
  `{obj}_reduction_vs_erm`, `{obj}_reduce30_seeds`); the **selection firewall** fields
  `source_only_reducers`, `best_reducer`, `confirmation_labels` (all chosen **source-only**),
  `final_task_preserving_reducers` (reported verdict, adds target retention),
  `gentle_gate_pass_source_only`, `gentle_gate_pass_with_target_retention`,
  `confirmation_label_selection_uses_target_eval=false`, `target_eval_used_for_verdict_only=true`; and
  the `n_perm_confirm` `confirmation` / `confirmation_per_seed` re-audit of the source-only labels.

## Rules

Target labels appear **only** in `target_eval` (evaluation-only); they never enter training,
normalization, the leakage audit, or baseline/config selection. The runner reports evidence and gate
verdicts but makes **no** A/B/C/D decision. Generated artifacts are exploratory; `commit_hash` +
`config_hash` pin each run.
