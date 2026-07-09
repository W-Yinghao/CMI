# W1 Alternative Split Rerun Feasibility

- status: GPU RERUN REQUIRED FOR CONFIRMATORY REPAIR
- h2cmi_trial_level_artifacts_available: `False`
- spdim_trial_level_artifacts_available: `False`
- can_recompute_h2cmi_without_gpu: `False`
- can_recompute_spdim_without_gpu: `False`
- requires_h2cmi_gpu_rerun: `True`
- requires_spdim_gpu_rerun: `True`
- expected_rows_h2cmi_if_rerun: `3450`
- expected_rows_spdim_seed0_if_rerun: `460`
- estimated_gpu_hours_h2cmi: `24.0`
- estimated_gpu_hours_spdim_seed0: `18.0`

The H2CMI estimate is a conservative planning placeholder because retained W1 artifacts do
not include per-target timing provenance; PM approval is still required before any rerun.

## Blockers

- H2CMI raw rows retain scalar metrics and prediction hashes, not trial-level predictions/logits.
- SPDIM P6 rows retain prediction/logits hashes, not trial-level prediction/logit arrays.
- Alternative split changes evaluation trial membership, so aggregate metrics cannot be recomputed from hashes.

## Red Team Review

- Hashes are not substitutes for trial-level predictions/logits.
- Changing the split changes the metric support, so old aggregate metrics cannot be recomputed exactly.
- This audit does not approve GPU work.
