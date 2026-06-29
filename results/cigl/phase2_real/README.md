# `results/cigl/phase2_real/` — Phase 2-real exploratory probe artifacts

Outputs of `scripts/run_cigl_phase2_real_probe.py` (see `docs/CIGL_10_PHASE2_REAL_PROBES.md`).

> **These are EXPLORATORY diagnostic artifacts, NOT benchmark results.** Every JSON carries
> `meta.exploratory = true` and `meta.setting = "strict_source_only_DG"`. They exist to inform the
> Gate-2 decision (does real source-only GraphCMINet-ERM leak label-conditional subject identity in
> its graph objects?). They are never a results table and must not be cited as accuracy/SOTA numbers.

The generated `*.json` and `*.npy` here are **gitignored** (regenerable run outputs); only this README
is tracked. Re-create them by running the script.

## Files

```
results/cigl/phase2_real/
  README.md                    # (tracked) this file
  <fold>_seed<S>.json          # one (fold, seed) audit: graph/node/edge leakage + null + split diagnostics
  <fold>_seed<S>_node_map.npy  # length-C per-channel leakage map  (regenerable)
  <fold>_seed<S>_edge_map.npy  # C×C per-edge binned-CMI map        (regenerable)
  <fold>_summary.json          # per-seed observed-vs-null rows + node/edge map seed-stability
```

`<fold>` is `synthetic` (dry-run) or `<dataset>_fold<F>` (real, e.g. `BNCI2014_001_fold0`).

## Per-seed JSON keys

- `meta`: `exploratory`, `setting`, `used_target_labels`, `used_target_covariates`, `dataset`,
  `fold`, `seed`, `n_perm`, `epochs`, `probe_epochs`, `n_classes`, `n_domains`, `commit_hash`,
  `config_hash`, `heldout_subject`.
- `source_info`: source subjects, encoder-train / probe-pool sizes, encoder-split diagnostics.
- `probe_split_diagnostics`: the support-aware (Y,D) split report (train/val domain support,
  `missing_val_domains`, `n_cells_low_support`, …).
- `graph` / `node` / `edge`: `kl_mean`, `permutation_{mean,std,p}`, `domain_acc`, `prior_acc`,
  `leakage_advantage`, `kl_ci`; `node` adds `node_leakage_map` (+ `_path`); `edge` adds
  `edge_leakage_top_k` (+ `edge_leakage_map_path`).

## `<fold>_summary.json`

`per_seed` observed-vs-null rows for graph/node/edge, plus `map_stability` (node/edge mean pairwise
correlation across seeds vs a random-map null). Read together with `docs/CIGL_10` §6 to choose Gate-2
path A/B/C/D.

## Provenance / reproducibility

`.npy` maps and `*.json` runs are gitignored generated artifacts; the JSON keeps compact inline
summaries so any committed record would be self-contained. `commit_hash` + `config_hash` pin each
run. The real run requires the offline MOABB datalake cache and should run on GPU/sbatch (CPU is
heavy); the synthetic dry-run is the binding engineering check.
