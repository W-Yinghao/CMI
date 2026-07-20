# C65 - Red-Team Verification

All C65 red-team gates pass.

- no_training_or_reinference_authorized: PASS - C65 remains recovery/readiness only.
- checkout_inventory_boundary_preserved: PASS - Checkout-only inventory remains incomplete.
- external_oaci_store_recovered: PASS - Mounted OACI frozen store is recovered by artifact index.
- non_oaci_weights_not_accepted: PASS - Adjacent S2P weights are not accepted as OACI frozen checkpoints.
- abi_validation_does_not_load_weights: PASS - No real checkpoint CPU load was attempted.
- preprocess_and_data_contract_recovered: PASS - Preprocess code and artifact-context data contract are recovered.
- mapping_complete: PASS - C50 singleton rows map to physical checkpoint files.
- trial_schema_ready: PASS - Trial cache schema is specified.
- split_label_guard_ready: PASS - Split-label grid forbids same-label reuse.
- full_cs_cache_missing: PASS - Full sample-level CS remains unsupported now.
- atom_trace_future_only: PASS - Atom trace requires future hooks or training instrumentation.
- reserved_holdout_preserved: PASS - Reserved dataset and seeds remain unused.
- mock_only_no_real_data: PASS - Mock ABI dry-runs use toy payloads only.
- forbidden_scan_passed: PASS - Forbidden affirmative claim scan passed.
- large_artifact_scan_passed: PASS - All listed artifacts are under 50MB.
- abi_no_torch_load_boundary_recorded: PASS - ABI metadata is sidecar-only; no torch load occurred.

## Slurm Validation

- focused_c65 job `890939` on `cpu-high` with `eeg2025`: `9 passed in 14.12s`.
- c50_c65_slice job `890940` on `cpu-high` with `eeg2025`: `170 passed in 6.76s`.
- c23_c65_regression job `890941` on `cpu-high` with `eeg2025`: `420 passed in 75.88s (0:01:15)`.
- full_oaci_tests job `890942` on `cpu-high` with `eeg2025`: `1344 passed in 429.94s (0:07:09)`.
