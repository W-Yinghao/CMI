# W1 Repaired H2CMI Dry-Run Audit

- dryrun_pass: `True`
- approve_gpu_run: `True`
- expected_rows: `3450`
- manifest_hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- datasets_passed: `['BNCI2014_001', 'Cho2017', 'Lee2019_MI']`
- datasets_blocked: `[]`
- all_eval_both_classes: `True`
- all_adapt_both_classes: `True`
- all_adapt_eval_disjoint: `True`
- target_label_leakage_detected: `False`
- method_selection_uses_target_performance: `False`

## Runner Dry-Run Crosscheck

- command: `python -m h2cmi.run_w1_repaired_p0 --dry-run --expected-rows 3450`
- path: `h2cmi/results/review_completion/w1_repaired_h2cmi_runner_dryrun_crosscheck.json`
- sha256: `8629e2f05969e9d128677b1d740256183a57e56d1426a5b73fed8c2de610f270`
- dryrun_pass: `True`
- approve_gpu_run: `True`
- manifest_units_checked: `345`
- expected_rows: `3450`

## Per Dataset

| dataset | manifest units | expected rows | eval both classes | adapt both classes | disjoint |
|---|---:|---:|---|---|---|
| BNCI2014_001 | 27 | 270 | `True` | `True` | `True` |
| Cho2017 | 156 | 1560 | `True` | `True` | `True` |
| Lee2019_MI | 162 | 1620 | `True` | `True` | `True` |

## Red Team Review

- No fitting, source-bundle load, model inference, Slurm submission, or GPU work occurred in P7A.
- Target labels are stored only as split-construction evidence through class counts, not as adaptation inputs.
- Dry-run approval only covers H2CMI repaired-split W1; it does not approve SPDIM or extra methods.
- Expected rows are bound to the P6.2 feasibility gate value 3450.
