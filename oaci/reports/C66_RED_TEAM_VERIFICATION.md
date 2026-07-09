# C66 - Red-Team Verification

All C66 red-team gates pass.

- authorization_phrase_absent_default_mode: PASS - No explicit micro-pilot authorization phrase was present.
- no_forward_or_reinference_executed: PASS - CPU torchload metadata only; no EEG forward.
- no_training_or_gpu: PASS - No training or GPU authorization.
- frozen_store_replayed: PASS - C65 frozen store recovery replayed.
- mapping_replay_complete: PASS - All grouped mapping rows verified.
- cpu_torchload_sample_passed: PASS - Six stratified checkpoints CPU-loaded successfully.
- state_hashes_match: PASS - Loaded state hashes match checkpoint ids.
- model_abi_no_forward: PASS - Model ABI validated without constructor/forward execution.
- preprocess_contract_validated: PASS - BNCI2014_001 preprocessing contract recovered.
- reserved_holdouts_preserved: PASS - BNCI2014_004 and seeds 3/4 remain reserved.
- no_cache_emitted: PASS - No real trial cache was emitted.
- split_label_claim_blocked: PASS - Few-label/split-label claims blocked without cache.
- full_cs_claim_blocked: PASS - Full conditional-CS claim blocked without paired samples.
- payload_policy_external_only: PASS - Large payloads remain external.
- forbidden_scan_passed: PASS - Forbidden affirmative claim scan passed.
- large_artifact_scan_passed: PASS - All C66 artifacts are under 50MB.

## Slurm Validation

- focused_c66 job `891232` on `cpu-high` with `eeg2025`: `8 passed in 0.21s`.
- c50_c66_slice job `891233` on `cpu-high` with `eeg2025`: `178 passed in 8.29s`.
- c23_c66_regression job `891234` on `cpu-high` with `eeg2025`: `428 passed in 73.06s (0:01:13)`.
- full_oaci_tests job `891235` on `cpu-high` with `eeg2025`: `1352 passed in 721.65s (0:12:01)`.
