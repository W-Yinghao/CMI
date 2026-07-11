# FP-GEM Smoke Audit

- status: `PASS`
- approve P12B fleet: `True`
- Slurm job: `893433`
- final squeue absence: `True`
- source state SHA-256: `dedf480348809008c79e822608e22de2787ddbbf1b6faa19bd9afb14c4abe1bd`
- source checkpoint file SHA-256: `cf295f0c8288db3543c0e42dc4d8ab5ea6abb1d681d8bd337718e43d5350dc4d`
- feature dimension: `210`
- maximum hook replay error: `0.0`
- evaluation labels accessed: `False`
- performance metrics computed: `False`
- stderr status: `empty`

The smoke validated only exact-config source retraining, persisted checkpoint provenance, six-method checkpoint identity, feature-hook semantics, shapes, finite transforms/logits, frozen parameters, and leakage boundaries. Accuracy and balanced accuracy were neither computed nor recorded and cannot influence the frozen P12 configuration.

## Gate Checks

- job_absent_from_squeue: `True`
- smoke_payload_pass: `True`
- exact_unit: `True`
- p9_reference_state_matches_committed_row: `True`
- source_reproduction_mode: `True`
- p9_checkpoint_file_unavailable: `True`
- actual_source_state_hash_complete: `True`
- checkpoint_file_checksum: `True`
- feature_dimension: `True`
- feature_hook_semantics: `True`
- six_method_prediction_shapes: `True`
- six_method_prediction_hashes_complete: `True`
- six_method_logits_hashes_complete: `True`
- six_methods_share_source_state: `True`
- p9_rows_not_reused: `True`
- no_performance_metrics: `True`
- evaluation_labels_not_accessed: `True`
- target_labels_not_passed: `True`
- no_target_selection: `True`
- classifier_frozen: `True`
- parameters_frozen: `True`
- fp_prior_fixed: `True`
- clean_launch: `True`
- runner_hash: `True`
- config_hash: `True`
- stdout_exists: `True`
- stdout_clean_header: `True`
- stderr_accepted: `True`
