# SPDIM W1 Repaired Seeds 1/2 Dry-Run Audit

- status: `PASS`
- approve_gpu_run: `True`
- launch_seeds: `[1, 2]`
- expected_rows_total: `920`
- manifest_hash_matches_p8: `True`
- runner_hash_matches_p8: `True`
- config_hash_matches_p8: `True`
- external_spdim_commit_matches: `True`
- worktree_clean_for_launch: `True`

## Split and Label Gates

- all_eval_both_classes: `True`
- all_adapt_both_classes: `True`
- all_adapt_eval_disjoint: `True`
- target_label_leakage_detected: `False`
- target_performance_selection_detected: `False`
- pretrained_weight_detected: `False`
- vendoring_detected: `False`
- slurm_accounting_script_calls_detected: `False`

## Actual CPU Dry-Run Runtime

| field | value |
|---|---|
| sys_executable | `/home/infres/yinwang/anaconda3/envs/icml/bin/python` |
| sys_prefix | `/home/infres/yinwang/anaconda3/envs/icml` |
| python_version | `3.9.25` |
| pytorch_version | `2.8.0+cu128` |
| cuda_version | `12.8` |
| device_name | `CPU` |
| moabb_version | `1.2.0` |
| mne_version | `1.8.0` |
| conda_default_env_inherited_label | `base` |

The dry-run device is CPU. GPU tasks must record their actual allocated device in stdout and shard summary. The inherited conda label is not used as environment proof.

## Dataset and Model Gate

| dataset | targets | seed models instantiated | tensor shape | expected new rows |
|---|---:|---:|---|---:|
| BNCI2014_001 | 9 | 2/2 | `[2592, 22, 500]` | 72 |
| Cho2017 | 52 | 2/2 | `[10520, 64, 500]` | 416 |
| Lee2019_MI | 54 | 2/2 | `[10800, 62, 500]` | 432 |

Exact source subjects, adaptation/evaluation trial IDs, class counts, and split hashes for both seeds and every target are retained in the JSON audit.

## Worktree Gate

The branch was clean at P9A start. During this audit, `worktree_clean_for_launch` allows only the named P9A controller, launcher, protocol/audit outputs, and command-log update pending their required commit. GPU launch still requires literal empty `git status --porcelain` after P9A is pushed.

## Red Team Review

- The P8 runner/config/manifest hashes are independently recomputed rather than inferred from filenames.
- Both approved seeds instantiate every dataset shape and build the frozen source-training command.
- The dry-run inspects target labels only for post-hoc split composition auditing; runtime adaptation loaders contain dummy labels only.
- No GPU job is approved if any required boolean is false.
