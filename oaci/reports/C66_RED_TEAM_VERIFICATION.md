# C66 - Red-Team Verification

All C66 red-team gates pass.

Authorization present: `True`. Authorized execution status: `executed`. External cache rows: `3456`.

- authorization_boundary_consistent: PASS - Authorization status matches C66 mode.
- forward_matches_authorization: PASS - Forward execution is either absent by default or present only after authorization.
- no_training_or_gpu: PASS - No training or GPU authorization.
- frozen_store_replayed: PASS - C65 frozen store recovery replayed.
- mapping_replay_complete: PASS - All grouped mapping rows verified.
- cpu_torchload_sample_passed: PASS - Six stratified checkpoints CPU-loaded successfully.
- state_hashes_match: PASS - Loaded state hashes match checkpoint ids.
- model_abi_consistent: PASS - Model ABI validated; authorized branch records strict CPU load/forward if executed.
- preprocess_contract_validated: PASS - BNCI2014_001 preprocessing contract recovered.
- reserved_holdouts_preserved: PASS - BNCI2014_004 and seeds 3/4 remain reserved.
- reserved_seed_sample_guard: PASS - Authorized/fallback ABI sample excludes reserved seeds 3/4.
- cache_emission_matches_authorization: PASS - Real cache is absent without authorization or external-only with authorization.
- content_addressed_cache_path: PASS - Authorized cache path is content-addressed by trial cache SHA prefix.
- cpu_runtime_guard: PASS - Authorized microcampaign records CPU-only/no-training execution.
- label_view_masking_contract: PASS - C66 label-view masking policy is emitted and bars selection-rule use.
- split_label_claim_blocked: PASS - Few-label/split-label claims remain blocked despite the cache foundation.
- full_cs_claim_blocked: PASS - Full conditional-CS claim remains blocked until a registered estimator is run.
- payload_policy_external_only: PASS - Large payloads remain external.
- external_manifest_hashes_present_if_cache_created: PASS - Authorized external cache artifacts are hash-manifested.
- authorized_cache_not_git_tracked: PASS - Authorized cache payload rows are external only.
- forbidden_scan_passed: PASS - Forbidden affirmative claim scan passed.
- large_artifact_scan_passed: PASS - All C66 artifacts are under 50MB.

## Slurm Validation

- bnci_real_preflight job `891248` on `cpu-high` with `eeg2025`: `PASS 12 bnci-real-preflight tests`.
- authorized_c66_generation_tightened job `891296` on `cpu-high` with `eeg2025`: `C66-A / REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED`.
- focused_c66_authorized_tightened job `891297` on `cpu-high` with `eeg2025`: `10 passed in 9.46s`.
- c50_c66_slice_authorized_tightened job `891299` on `cpu-high` with `eeg2025`: `180 passed in 62.35s (0:01:02)`.
- c23_c66_regression_authorized_tightened job `891300` on `cpu-high` with `eeg2025`: `430 passed in 62.35s (0:01:02)`.
- full_oaci_tests_authorized_tightened job `891301` on `cpu-high` with `eeg2025`: `1354 passed in 298.11s (0:04:58)`.
