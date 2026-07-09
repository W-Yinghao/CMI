# SPDIM W1 Repaired-Split Seed-0 Dry-Run Audit

- status: `PASS`
- approve_gpu_run: `True`
- manifest_hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- manifest_hash_matches_p7: `True`
- expected_rows_total: `460`
- worktree_clean_for_launch: `True`
- external_spdim_commit: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`

## Gate Checks

- all_eval_both_classes: `True`
- all_adapt_both_classes: `True`
- all_adapt_eval_disjoint: `True`
- target_label_leakage_detected: `False`
- pretrained_weight_detected: `False`
- vendoring_detected: `False`
- shape_blocker_detected: `False`

## Dataset Summary

| dataset | targets | expected rows | adapt counts | eval counts | tensor shape |
|---|---:|---:|---|---|---|
| BNCI2014_001 | 9 | 36 | `[(36, 36)]` | `[(36, 36)]` | `[2592, 22, 500]` |
| Cho2017 | 52 | 208 | `[(50, 50), (60, 60)]` | `[(50, 50), (60, 60)]` | `[10520, 64, 500]` |
| Lee2019_MI | 54 | 216 | `[(25, 25)]` | `[(25, 25)]` | `[10800, 62, 500]` |

## Split Evidence

Exact source subject IDs, adaptation trial IDs, evaluation trial IDs, and split hashes for every target subject are recorded in the JSON audit.

## Red Team Review

- The dry-run does not train or adapt on GPU.
- The expensive official SPD forward check is optional and defaults to zero samples; the required gate is model instantiation plus loader/manifest validation.
- Target adaptation loaders use dummy labels; target labels are not available to adaptation or method selection.
- `worktree_clean_for_launch` is evaluated after P7 cache hygiene allowing only expected P8A files pending commit; the GPU launch still requires an actually clean post-commit worktree.
