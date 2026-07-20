# C67 - Red-Team Verification

All C67 red-team gates pass.

- c66_dual_mode_resolved: PASS - C66 no-auth guard and authorized cache are explicitly separated.
- cache_hashes_match: PASS - External cache and manifest hashes match committed C66 manifest.
- cache_integrity_passed: PASS - Trial cache schema/numeric integrity gates pass.
- masking_contract_passed: PASS - C66 label-view projections mask labels/predictions as required.
- no_new_forward_training_gpu: PASS - C67 consumed cache only.
- reserved_holdouts_preserved: PASS - No BNCI2014_004 or seeds 3/4.
- split_label_not_sufficiency: PASS - Split-label smoke is underpowered and not a sufficiency claim.
- cs_not_full_claim: PASS - Conditional-CS result is a proxy smoke and underpowered.
- endpoint_oracle_boundary_preserved: PASS - Same-label oracle remains diagnostic-only.
- diagnostic_full_views_policy_only: PASS - Diagnostic/oracle full views are explicitly policy-only and unavailable for selection.
- large_artifact_scan_passed: PASS - All C67 git artifacts are below 50MB.
- forbidden_scan_passed: PASS - Forbidden affirmative claim scan passed.

## Slurm Validation

- focused_c67 job `891382` on `cpu-high` with `eeg2025`: `8 passed in 0.24s`.
- c50_c67_slice job `891383` on `cpu-high` with `eeg2025`: `188 passed in 16.36s`.
- c23_c67_regression job `891387` on `cpu-high` with `eeg2025`: `438 passed in 46.27s`.
- full_oaci_tests job `891384` on `cpu-high` with `eeg2025`: `1362 passed in 289.26s`.
