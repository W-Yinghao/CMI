# FP-GEM Final Red-Team Review

Status: `PASS`.

This review independently parsed the final CSV and raw unit payloads, rebuilt the seed-averaged subject table, reran the frozen 10,000-replicate paired cluster bootstrap, and compared every method estimate and contrast interval with the analyzer output.

## Gate Results

- result_rows_1134: `True`
- unique_result_keys: `True`
- unit_count_189: `True`
- source_seeds_exact: `True`
- method_set_exact: `True`
- dataset_subject_counts: `True`
- all_rows_ok: `True`
- hashes_complete: `True`
- no_target_label_leakage: `True`
- no_target_performance_selection: `True`
- backbone_and_classifier_frozen: `True`
- result_origins_disclosed: `True`
- six_methods_share_each_source_state: `True`
- six_methods_share_each_checkpoint_file: `True`
- result_checksum_matches_summary: `True`
- seed_average_first: `True`
- per_subject_file_recomputes: `True`
- method_points_and_cis_recompute: `True`
- contrast_points_and_cis_recompute: `True`
- bootstrap_policy_exact: `True`
- raw_unit_count_189: `True`
- raw_split_and_leakage_gates: `True`
- raw_exact_p9_config_retrain_disclosed: `True`
- raw_fp_prior_fixed: `True`
- raw_classifier_frozen: `True`
- raw_runner_config_manifest_frozen: `True`
- artifact_manifest_all_pass: `True`
- eight_result_tasks_complete: `True`
- excluded_attempts_zero_accepted_rows: `True`
- final_squeue_absent: `True`
- claim_gate_expected: `True`

## Adversarial Findings

- P9 checkpoint files were unavailable; this packet uses exact-P9-configuration retrains, and all six methods share one persisted source checkpoint within each target-seed unit.
- The actual P12 source-state hashes do not match the P9 reference hashes, so direct P9 checkpoint or direct committed-row reuse must not be claimed.
- FP-GEM improves over source-only, but its subject-weighted contrast is negative against RCT and both SPDIM variants.
- FP-GEM minus Joint-GEM is small and its subject-weighted interval crosses zero; a general fixed-prior superiority claim is not supported.
- No equivalence, noninferiority, broad-benchmark, or target-tuned claim is supported.

## Claim Gate

- fp_gem_improves_over_source_only_supported: `True`
- fp_gem_improves_over_rct_supported: `False`
- fp_gem_improves_over_spdim_geodesic_supported: `False`
- fp_gem_improves_over_spdim_bias_supported: `False`
- fp_gem_improves_over_joint_gem_subject_weighted_supported: `False`
- equivalence_or_noninferiority_supported: `False`
- broad_benchmark_claim_supported: `False`
- direct_p9_checkpoint_reuse_claim_supported: `False`

Final result SHA-256: `f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d`.
