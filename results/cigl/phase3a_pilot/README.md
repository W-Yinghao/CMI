# `results/cigl/phase3a_pilot/` — Phase 3A regularizer-effect pilot artifacts

Outputs of `scripts/run_cigl_phase3a_regularizer_pilot.py` (see `docs/CIGL_14_PHASE3A_REGULARIZER_PILOT.md`).

> **EXPLORATORY pilot artifacts, NOT benchmark results.** Every JSON carries `meta.exploratory = true`,
> `meta.phase = "Phase3A_regularizer_effect_pilot"`, and `meta.setting = "strict_source_only_DG"`. They
> test whether GraphCMI regularization can reduce leakage without destroying task performance on one
> dataset / one fold — they are not a results table and must not be cited as accuracy/SOTA numbers.

The generated `*.json` here are **gitignored** (regenerable run outputs); only this README is tracked.

## Files

```
results/cigl/phase3a_pilot/
  README.md                                                       # (tracked) this file
  <dataset>_fold<F>_<config>_seed<S>_nperm<N>.json                # Pass 1 (all 7 configs, n_perm=20)
  <dataset>_fold<F>_confirm_<config>_seed<S>_nperm<N>.json        # Pass 2 (erm/full_cigl/best-Pareto, n_perm=50)
  <dataset>_fold<F>_phase3a_summary.json                          # per-config means + confirmation + per-seed
```

`<config>` ∈ {erm, graph_only, node_only, edge_only, graph_node, full_cigl, low_full_cigl}. **Pass 1** =
all 7 configs at `n_perm=20`; **Pass 2** = ERM / full_cigl / best-Pareto re-audited at `n_perm=50` (same
frozen model, higher permutation power) with **per-seed confirmation records retained**. Confirmation
leakage reductions are computed against the **confirmation ERM** (`confirm_leakage_reduction_vs_confirm_erm`);
pass-1 reductions are vs the pass-1 ERM (`pass1_leakage_reduction_vs_erm`). best-Pareto is selected from
**source-only** metrics and **full_cigl is eligible**. Target labels appear **only** in `target_eval`.

## Per (config, seed) JSON

- `config`, `graphcmi`, `lambda_g/node/edge`, `seed`, `n_perm`, `gate_alpha`, `heldout_subject`.
- `source_probe`: `balanced_acc`, `macro_f1` (held-out source probe-pool).
- `target_eval`: `balanced_acc`, `macro_f1`, **`evaluation_only: true`** (target labels used only here).
- `graph`/`node`/`edge`: `kl_mean`, `permutation_mean`, `permutation_p`, `positive_excess`,
  `clears_null`, `gate_alpha` (fresh held-out probes; retrained within-train permutation null).
- `stepA`: `graph/node/edge_dom_acc`, `*_loss`, `reg_graph/node/edge` (training-head diagnostics — NOT
  leakage evidence).
- `probe_split_diagnostics`, `n_enc_train`, `n_probe_pool`, `meta`.

## `<dataset>_fold<F>_phase3a_summary.json`

- `meta` (exploratory/phase/setting + target-label-rule flags + commit/config hash).
- `erm_reference_kl` (graph/node/edge ERM KL baseline).
- `best_pareto_config` (chosen from **source-only** metrics; never from target_eval).
- `per_config`: per-config means — `source_probe_bacc/f1`, `target_eval_bacc/f1`,
  `{graph,node,edge}_kl_mean`, `{...}_leakage_reduction_vs_erm`, `{...}_clears_null_count`.
- `confirmation`: ERM / full_cigl / best-Pareto re-audited at `n_perm_confirm` (=50) with
  `{...}_clears_null_count_nperm50`.

Read with `docs/CIGL_14` §pass/fail to choose next path A (full CIGL) / B (graph+node) / C (node-only)
/ D (diagnostic-only). Generated artifacts are exploratory; `commit_hash` + `config_hash` pin each run.
