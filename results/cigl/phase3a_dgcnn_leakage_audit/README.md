# results/cigl/phase3a_dgcnn_leakage_audit/ — Phase 3A-H outputs

**EXPLORATORY diagnostic outputs** from `scripts/run_cigl_phase3a_dgcnn_leakage_audit.py` (CIGL Phase
3A-H; see `docs/CIGL_24_PHASE3A_H_DGCNN_LEAKAGE_AUDIT.md`). Graph/node leakage audit of the task-capable
`dgcnn_forward_graph_adapter` under the **same strict source-only** BNCI2014_001 fold-0, **ERM only (no
CMI regularization)**. The **edge audit is skipped** (DGCNN has static/shared adjacency, no per-sample
edge object — never faked). Generated `*.json` are **gitignored**; the tracked record is `docs/CIGL_25`.

## Files

- `<fold>_dgcnn_forward_graph_adapter_seed<k>.json` — per seed: `train` / `source_probe` /
  `target_eval` (`balanced_acc`, `macro_f1`; `target_eval.evaluation_only=true`),
  `train_minus_source_gap`, `graph_usage` (`zero_graph`/`permute_nodes` bAcc + drops, `graph_path_used`),
  `leakage` = `{graph, node}` blocks (`kl_mean`, `permutation_mean`, `permutation_p`, `clears_null`;
  node adds `node_leakage_map`), `edge_audit_skipped=true` + `edge_skip_reason`, and source-only `meta`.
- `<fold>_dgcnn_leakage_audit_summary.json` — `task` (source/train/target means, per-seed,
  `n_seeds_source_pass`, `task_ok`), `graph_usage`, `leakage` (graph/node `kl_mean` + `clears_null_seeds`),
  `node_map_stability` (`mean_corr`, `null_q95`, `above_random`, `degenerate`), the booleans
  `graph_leakage_exists` / `node_leakage_exists` / `leakage_exists` / `audit_passes`, `edge_skip_reason`,
  and firewall meta (`used_target_labels_for_{training,selection}=false`, `used_target_covariates=false`,
  `target_eval_is_evaluation_only=true`, `cmi_regularization_used=false`, `edge_audit_skipped=true`).

## Reading the result

- graph/node leakage clears the null (≥2/3 seeds) on a task-capable DGCNN → **A** (graph/node
  regularizer pilot may be considered — not edge-CMI).
- nothing clears the null → **B** (pause method path; diagnostic story only).
- DGCNN task fails on rerun (source < 0.45) → **C** (return to backbone diagnosis).

`clears_null = kl_mean > permutation_mean AND permutation_p ≤ gate_alpha (0.05)`, `n_perm=50` on the real
run. Each summary embeds `meta.commit_hash` / `meta.config_hash`. Exploratory diagnostic (one dataset,
one fold) — not a benchmark/SOTA table; feeds the Gate-3A-H decision only.
