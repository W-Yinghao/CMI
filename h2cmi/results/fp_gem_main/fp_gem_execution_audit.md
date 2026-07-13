# FP-GEM Execution Audit

- status: `PASS`
- accepted target-seed tasks: `189`
- new-method rows: `378`
- same-checkpoint official-control rows: `756`
- directly reused P9 rows: `0`
- final rows: `1134`
- final squeue absence: `True`
- stdout statuses: `{'exists_complete_clean_launch': 189}`
- stderr statuses: `{'empty': 189}`

## Accepted Jobs

- smoke: `{'job_id': '893433', 'launch_commit': '5b71ee841384327ad01e02f57f49378285266195', 'result_rows': 0, 'smoke_payload_sha256': '4bdcbb27f7303bc99f642119ae996b936c4a77b56a026c393e729bc87c672fe7', 'status': 'pass_no_performance', 'submit_command': 'sbatch --parsable -p V100 --export=ALL,FP_GEM_REPO=/home/infres/yinwang/CMI_AAAI/.codex_p12_launch_5b71ee8 scripts/fp_gem_smoke.slurm'}`
- array: `{'accepted_result_tasks': ['1', '2', '3', '4', '5'], 'array': '0-5%6', 'hardware_group': 'V100', 'job_id': '893448', 'launch_commit': 'e6c491560c14ae44cbb2bd03e0d8c1214e07da56', 'result_rows': 804, 'scientific_configuration_changed': False, 'source_units': 134, 'submit_command': 'sbatch --parsable -p V100 --array=0-5%6 --export=ALL,FP_GEM_REPO=/home/infres/yinwang/CMI_AAAI/.codex_p12_launch_5b71ee8,FP_GEM_GROUP=V100,FP_GEM_GROUP_COUNT=161,FP_GEM_GROUP_STRIDE=6 scripts/fp_gem_array.slurm', 'working_directory': '/home/infres/yinwang/CMI_AAAI/.codex_p12_launch_5b71ee8'}`
- array: `{'accepted_result_tasks': ['0', '1'], 'array': '0-1%2', 'hardware_group': 'A100', 'job_id': '893453', 'launch_commit': 'e6c491560c14ae44cbb2bd03e0d8c1214e07da56', 'result_rows': 168, 'scientific_configuration_changed': False, 'source_units': 28, 'submit_command': 'sbatch --parsable -p A100 --array=0-1%2 --export=ALL,FP_GEM_REPO=/home/infres/yinwang/CMI_AAAI/.codex_p12_launch_5b71ee8,FP_GEM_GROUP=A100,FP_GEM_GROUP_COUNT=28,FP_GEM_GROUP_STRIDE=2 scripts/fp_gem_array.slurm', 'working_directory': '/home/infres/yinwang/CMI_AAAI/.codex_p12_launch_5b71ee8'}`
- array: `{'accepted_result_tasks': ['0'], 'array': '0-0%1', 'hardware_group': 'V100', 'job_id': '893456', 'launch_commit': 'e6c491560c14ae44cbb2bd03e0d8c1214e07da56', 'result_rows': 162, 'scientific_configuration_changed': False, 'source_units': 27, 'submit_command': 'sbatch --parsable -p V100 --array=0-0%1 --export=ALL,FP_GEM_REPO=/home/infres/yinwang/CMI_AAAI/.codex_p12_launch_5b71ee8,FP_GEM_GROUP=V100,FP_GEM_GROUP_COUNT=161,FP_GEM_GROUP_STRIDE=6 scripts/fp_gem_array.slurm', 'working_directory': '/home/infres/yinwang/CMI_AAAI/.codex_p12_launch_5b71ee8'}`

## Excluded Zero-Result Launches

- excluded: `{'accepted_rows': 0, 'job_id': '893415', 'reason': 'compute node could not see the pre-amendment temporary worktree', 'status': 'zero_result_infrastructure_failure'}`
- excluded: `{'accepted_rows': 0, 'job_id': '893416', 'reason': 'overstrict unrecoverable P9 byte-hash gate stopped before RCT, GEM, evaluation labels, or performance', 'status': 'zero_result_pre_amendment_gate_stop'}`
- excluded: `{'accepted_rows': 0, 'array_job_id': '893448', 'array_task_id': '0', 'failed_payload_sha256': '2dca2cab184c00f6a696c9b1ca8f64d5136562242808c4cf53cbf9cba3162662', 'job_id': '893449', 'reason': 'cached smoke checkpoint reload omitted custom SPD batch-normalization source-domain running buffers', 'retry_job_id': '893456', 'stderr_sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'stdout_sha256': 'd311a2fde4d5b28c65a0f67a9fc8d4c66a3fcbfee2dce3a6797ee27356123082', 'status': 'failed_attempt_excluded'}`
- excluded: `{'accepted_rows': 0, 'job_created': False, 'reason': 'QOSMaxSubmitJobPerUserLimit', 'requested_array': '0-160%6', 'status': 'scheduler_rejected_before_job_creation'}`
- excluded: `{'accepted_rows': 0, 'job_created': False, 'reason': 'QOSMaxSubmitJobPerUserLimit', 'requested_array': '0-63%6', 'status': 'scheduler_rejected_before_job_creation'}`
- excluded: `{'accepted_rows': 0, 'job_created': False, 'reason': 'QOSMaxSubmitJobPerUserLimit', 'requested_array': '0-31%6', 'status': 'scheduler_rejected_before_job_creation'}`

Completion uses only `squeue` absence plus stdout/stderr and artifact parse/count/checksum validation. No Slurm accounting command is used.

## Scientific Gates

- accepted_unit_count: `189`
- new_method_row_count: `378`
- within_unit_control_row_count: `756`
- reused_p9_row_count: `0`
- final_row_count: `1134`
- expected_new_method_rows: `378`
- expected_within_unit_control_rows: `756`
- expected_reused_p9_rows: `0`
- expected_final_rows: `1134`
- duplicate_keys: `0`
- all_rows_ok: `True`
- prediction_hashes_complete: `True`
- logits_hashes_complete: `True`
- all_adapt_eval_disjoint: `True`
- all_eval_both_classes: `True`
- target_label_leakage_detected: `False`
- target_performance_selection_detected: `False`
- all_sources_exact_p9_config_retrains: `True`
- p9_reference_state_hash_match_count: `0`
- all_six_methods_share_unit_source_state: `True`
- all_classifiers_frozen: `True`
- manifest_hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- source_seeds: `[0, 1, 2]`
- squeue_absence: `True`
- stdout_validation_pass: `True`
- stderr_validation_pass: `True`
- validation_pass: `True`
