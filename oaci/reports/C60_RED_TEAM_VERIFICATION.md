# C60 - Red-Team Verification

All C60 red-team gates pass.

- proof_audit_completed: PASS - Line-item proof audit rows are emitted.
- proof_repaired_not_overclaimed: PASS - Multi-candidate/top-k material is demoted unless assumptions are added.
- bridge_partial_not_full: PASS - Empirical bridge is partial and has missing theorem-critical data.
- source_escape_hatch_closed: PASS - No source-observable counterexample is found.
- lecam_not_upgraded: PASS - Le Cam remains empirical for EEG artifacts.
- fano_not_upgraded: PASS - Fano/Assouad remains blocked.
- endpoint_oracle_diagnostic_only: PASS - Same-label endpoint scalar remains unavailable at selection time.
- training_not_executed: PASS - C60 refines blueprint but does not execute training.
- reserved_dataset_and_seeds_blocked: PASS - BNCI2014_004 and seeds [3,4] remain reserved.
- no_gpu_or_reinference: PASS - No GPU or re-inference is authorized.
- no_m1_or_manuscript: PASS - C60 does not start M1 or manuscript drafting.
- no_eeg_theorem_claim: PASS - Rank-gauge theorem remains synthetic/model-bound.
- forbidden_scan_passed: PASS - Forbidden affirmative claim scan passed.
- large_artifact_scan_passed: PASS - All listed artifacts are under 50MB.
- no_selector_artifact: PASS - C60 emits no selected-candidate or chosen-checkpoint artifact.
- test_manifest_recorded: PASS - Validation scopes are recorded for Slurm cpu-high.
