# FP-GEM Prevalence Checkpoint-Reuse Gate

- status: `PENDING_PRECOMMITTED_PROBE`
- approved action: one V100 shape/numerical/checkpoint/hash gate on `Lee2019_MI`, target 1, source seed 0
- performance metrics computed: `false`
- full fleet approved: `false`
- fresh source training permitted: `false`

The gate command, job ID, runtime environment, stdout/stderr checksums, and exact P12 hash comparisons will be appended after the precommitted probe leaves `squeue`. Any mismatch produces a blocker and prevents fleet launch.
