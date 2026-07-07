# S2P 9B-0 SLURM smoke report

jobs 888763 (cbramod N32), 888764 (cbramod N128), 888765 (codebrain_stage2 N32); partition A40 GPU.
run_id key = model_condition_N{n}_H{hours}_s{seed} (joins to model_matrix/pretrain_subset_design).
artifacts: ckpt_<run_id>.pt, smoke_<run_id>.json per run. resume/failure: standard SLURM; det pinned (CUBLAS_WORKSPACE_CONFIG, use_deterministic_algorithms warn_only).

| run_id | pass | ckpt | loss(first->last) |
|---|---|---|---|
| cbramod_fixed_hours_N128_H50_s0 | True | True | [16.53352, 2.34646] |
| cbramod_fixed_hours_N32_H50_s0 | True | True | [16.58449, 2.24299] |
| codebrain_stage2_fixed_hours_N32_H50_s0 | False | None | None |
