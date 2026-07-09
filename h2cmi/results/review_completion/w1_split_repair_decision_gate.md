# W1 Split Repair Decision Gate

- legacy_w1_quarantined: `True`
- valid_subset_recomputed: `True`
- alternative_split_found: `True`
- alternative_split_all_datasets_pass: `True`
- approve_h2cmi_alternative_split_rerun: `False`
- approve_spdim_alternative_split_seed0_rerun: `False`
- approve_spdim_seeds_1_2: `False`
- approve_full_spdim: `False`
- next_gpu_step_requires_pm_approval: `True`

## Recommended Split

`class_stratified_half` passes all datasets in dry-run, but it is not approved for execution in this step.

## Red Team Review

- Legacy W1/SPDIM are quarantined for confirmatory use.
- Valid-subset recompute is diagnostic-only and excludes Cho2017.
- A replacement split is designed but no rerun is approved.
- No seeds 1/2 or full SPDIM expansion are approved.
