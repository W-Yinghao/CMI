# results/cigl/phase3a_dgcnn_gn_regularizer_pilot/ — Phase 3A-I outputs

**EXPLORATORY pilot outputs** from `scripts/run_cigl_phase3a_dgcnn_gn_regularizer_pilot.py` (CIGL Phase
3A-I; see `docs/CIGL_26_PHASE3A_I_DGCNN_GN_REGULARIZER_PILOT.md`). Graph/node CMI regularizer pilot on the
task-capable `dgcnn_forward_graph_adapter`, **strict source-only** BNCI2014_001 fold-0, **graph/node only
(no edge term)**. Generated `*.json` are **gitignored**; the tracked record is `docs/CIGL_27`.

## Files

- `<fold>_<config>_seed<k>.json` — per config × seed: `lambda_g`/`lambda_node`, `method` (`erm` or
  `graphcmi`), `train`/`source_probe`/`target_eval` (`balanced_acc`,`macro_f1`;
  `target_eval.evaluation_only=true`), `train_minus_source_gap`, `graph_usage`
  (`zero_graph`/`permute_nodes` drops, `graph_path_used`), `leakage`={`graph`,`node`} blocks
  (`kl_mean`,`permutation_mean`,`permutation_p`,`clears_null`), `edge_audit_skipped=true` +
  `edge_skip_reason`, and source-only `meta` flags.
- `<fold>_dgcnn_gn_pilot_summary.json` — `configs` (λ map), `per_config` aggregates (source per-seed,
  graph/node `kl_mean` + per-seed, `*_clears_seeds`, graph-usage), `selection` (the **source-only**
  firewall: `reductions`, `erm_reproduces`, `source_only_reducers`, `best_pareto`, `best_graph_node`,
  `confirmation_labels`, `final_target_retaining_reducers`, `pilot_pass_source_only`,
  `pilot_pass_with_target_retention`), `confirmation` (n_perm=50 re-audit of the source-only confirmation
  labels), and `edge_skip_reason`.

## Reading the result

- `erm_fixed` must reproduce (source ≥0.45, ≥2/3 seeds) else **Decision D** (DGCNN stability).
- a config with ≥30% graph/node KL reduction (≥2/3 seeds), source drop ≤0.02, source ≥0.45, target drop
  ≤0.05 → **Decision A** (candidate for multi-fold confirmation).
- source retained but target/headroom thin → **Decision B (tradeoff)** — not a method win.
- no reduction without source loss → **Decision C** (diagnostic / redesign).

`clears_null = kl_mean > permutation_mean AND permutation_p ≤ gate_alpha (0.05)`. Selection and
confirmation labels are **source-only**; `target_eval` is evaluation-only (reported retention verdict
only). **No edge term / audit** (DGCNN static adjacency). Exploratory pilot (one dataset, one fold) — not
a benchmark/SOTA table.
