# FP-GEM Prevalence-Stress Execution Audit

- status: `PASS`
- accepted target-seed units: `162`
- final rows: `2916`
- geometry rows: `972`
- final squeue absence: `True`
- stdout statuses: `{'exists_complete_clean_launch': 162}`
- stderr statuses: `{'empty': 161, 'verified_post_artifact_scheduler_handoff': 1}`
- submission-record snapshot SHA-256: `eac311f6d87da554d22734b9d43a949b0511be338772b51848395601b6efc2d5`

## Accepted Result-Carrying Jobs

- checkpoint gate: `{'artifact_sha256': '4c8ef4631d6bcfe09d2d13443238c146b206f6bd1c07c8933b72be855f8b1535', 'command': 'sbatch --parsable --partition=V100 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_7b48813 scripts/fp_gem_prevalence_checkpoint_gate.slurm', 'job_id': '894726', 'status': 'pass'}`
- job `894784`: accepted units `50`, hardware groups `['V100']`
- job `894790`: accepted units `28`, hardware groups `['A100']`
- job `894879`: accepted units `1`, hardware groups `['V100']`
- job `894897`: accepted units `21`, hardware groups `['V100']`
- job `894903`: accepted units `20`, hardware groups `['V100']`
- job `894918`: accepted units `21`, hardware groups `['V100']`
- job `894921`: accepted units `21`, hardware groups `['V100']`

## Failed And Excluded Attempts

- `{"accepted_result_rows": 0, "array_task_id": "4", "failure_reason": "P13 source density does not reproduce P12", "job_id": "894784", "raw_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task4/failed_Lee2019_MI_target2_seed1.json", "raw_sha256": "4f6bf732a15e8de69d623a98f5eff540e99f2f8af3cc2d0aff007df9b03ba739", "source_seed": 1, "status": "verified_excluded", "stderr_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task4/894784_4.err", "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "stdout_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task4/894784_4.out", "stdout_sha256": "7b6edc179735b001bef6204223d17780c726219c1d7c206578936016b08d841b", "target_subject": 2}`
- `{"accepted_result_rows": 0, "array_task_id": "2", "failure_reason": "P13 source density does not reproduce P12", "job_id": "894784", "raw_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task2/failed_Lee2019_MI_target3_seed2.json", "raw_sha256": "381dc8b1377717731b03683686c044e9ac858e463f173bc2cb03b658df2e09b7", "source_seed": 2, "status": "verified_excluded", "stderr_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task2/894784_2.err", "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "stdout_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task2/894784_2.out", "stdout_sha256": "9d2c652045a893dcc349411d921e9218e087aa50319f12e86e913d75ac77c56b", "target_subject": 3}`
- `{"accepted_result_rows": 0, "array_task_id": "4", "failure_reason": "P13 source density does not reproduce P12", "job_id": "894841", "raw_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894841_task4/failed_Lee2019_MI_target2_seed1.json", "raw_sha256": "30a5343e9119ed60687172b58bbafd7468c202be7dcfb8e5408a8b0b7c549146", "source_seed": 1, "status": "verified_excluded", "stderr_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894841_task4/894841_4.err", "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "stdout_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894841_task4/894841_4.out", "stdout_sha256": "02dc575118aa7e5df637fb1d7fe1db2f17b0be53b3dd9d7907c7af659500636d", "target_subject": 2}`
- `{"accepted_result_rows": 0, "array_task_id": "2", "failure_reason": "P13 source density does not reproduce P12", "job_id": "894863", "raw_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894863_task2/failed_Lee2019_MI_target3_seed2.json", "raw_sha256": "ad29cb0ae3bc1b14c875967f07917758c804846b090b69c7913e9805ae4a28cd", "source_seed": 2, "status": "verified_excluded", "stderr_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894863_task2/894863_2.err", "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "stdout_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894863_task2/894863_2.out", "stdout_sha256": "8c612cf4768bb4660aed7f121cade33afc1aeda5b2700e12904343bef2e85ac0", "target_subject": 3}`
- `{"accepted_result_rows": 0, "array_task_id": "3", "failure_reason": "P13 source density does not reproduce P12", "job_id": "894784", "raw_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task3/failed_Lee2019_MI_target6_seed0.json", "raw_sha256": "1e96ff24bede46bcd769d15a99bed2ce4225fe6dee17916521eac4816d986ef8", "source_seed": 0, "status": "verified_excluded", "stderr_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task3/894784_3.err", "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "stdout_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task3/894784_3.out", "stdout_sha256": "bd88fbe106189544ce9be8af24d1f8f1e9cfb4f1084930012347fb61d0a0ca18", "target_subject": 6}`
- `{"accepted_result_rows": 0, "array_task_id": "0", "failure_reason": "P13 source density does not reproduce P12", "job_id": "894784", "raw_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task0/failed_Lee2019_MI_target5_seed0.json", "raw_sha256": "ea337125485869b1b34a61c984e3e5e9984e3848d85c6221c794bd1d9078e456", "source_seed": 0, "status": "verified_excluded", "stderr_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task0/894784_0.err", "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "stdout_path": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/excluded/894784_task0/894784_0.out", "stdout_sha256": "be442266299709333981dfadabee48549bc5b6a7b5bd849b4c61d28136105811", "target_subject": 5}`

## Canceled Zero-Result Launches

- `{"accepted_rows": 0, "array_range": "4-4", "command": "sbatch --parsable --partition=V100 --nodelist=node42 --array=4-4 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm", "concurrency": 1, "group": "V100", "group_count": 134, "job_id": "894872", "node_constraint": "node42", "role": "exact_failed_task_hardware_reproduction_retry", "status": "canceled_pending_zero_result", "stride": 6}`
- `{"accepted_rows": 0, "array_range": "3-3", "command": "sbatch --parsable --partition=V100 --nodelist=node10 --array=3-3 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm", "concurrency": 1, "group": "V100", "group_count": 134, "job_id": "894902", "node_constraint": "node10", "role": "exact_failed_task_hardware_reproduction_retry", "status": "canceled_pending_zero_result", "stride": 6}`

## Verified Post-Artifact Scheduler Handoff

- `{"accepted_unit_keys": ["Lee2019_MI:2:1"], "array_range": "4-4", "command": "sbatch --parsable --partition=V100 --nodelist=node11 --array=4-4 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm", "concurrency": 1, "group": "V100", "group_count": 134, "job_id": "894879", "node_constraint": "node11", "role": "exact_failed_task_hardware_reproduction_retry", "status": "completed_repair_then_canceled_for_scheduler_handoff", "stride": 6}`
- The named unit was atomically complete before cancellation. The interrupted next group supplied zero accepted rows from that attempt and was completed by the frozen resume job. This stderr is not classified as a harmless warning.

## Effective Submission Commands

- job `894784`: `sbatch --parsable --partition=V100 --array=0-5 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894790`: `sbatch --parsable --partition=A100 --array=0-1 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=A100,FP_GEM_P13_GROUP_COUNT=28,FP_GEM_P13_GROUP_STRIDE=2 scripts/fp_gem_prevalence_array.slurm`
- job `894841`: `sbatch --parsable --partition=V100 --array=4-4 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894863`: `sbatch --parsable --partition=V100 --array=2-2 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894872`: `sbatch --parsable --partition=V100 --nodelist=node42 --array=4-4 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894879`: `sbatch --parsable --partition=V100 --nodelist=node11 --array=4-4 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894897`: `sbatch --parsable --partition=V100 --nodelist=node11 --array=2-2 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894902`: `sbatch --parsable --partition=V100 --nodelist=node10 --array=3-3 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894903`: `sbatch --parsable --partition=V100 --nodelist=node15 --array=3-3 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894918`: `sbatch --parsable --partition=V100 --array=4-4 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`
- job `894921`: `sbatch --parsable --partition=V100 --nodelist=node43 --array=0-0 --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 scripts/fp_gem_prevalence_array.slurm`

Completion uses job absence from `squeue` plus stdout, stderr, artifact parse/count, and checksum gates. No Slurm accounting command is used.

## Validation

- accepted_unit_count: `162`
- final_result_rows: `2916`
- geometry_rows: `972`
- duplicate_result_keys: `0`
- duplicate_geometry_keys: `0`
- all_rows_ok: `True`
- prediction_hashes_complete: `True`
- logits_hashes_complete: `True`
- checkpoint_hashes_complete: `True`
- adaptation_manifest_hashes_complete: `True`
- fixed_batch_size: `True`
- balanced_evaluation: `True`
- q05_exact_p12_center_count: `972`
- q05_all_exact: `True`
- source_only_endpoint_reuse_count: `324`
- source_only_q_independent: `True`
- new_adaptation_row_count: `1620`
- target_label_leakage_detected: `False`
- q_passed_to_method_detected: `False`
- target_performance_selection_detected: `False`
- fresh_source_training_detected: `False`
- all_adapt_eval_disjoint: `True`
- all_checkpoint_hashes_exact_p12: `True`
- all_runner_hashes_identical: `True`
- all_launch_commits_identical: `True`
- manifest_sha256: `8c5b160fcec5ffeaded7faaf196f9753d7e0f7f15e583f8a18a5651ddf1c5802`
- manifest_semantic_sha256: `29febb846ab5935dfed398953b28cbc2da86862842edf4c851a21515df71263f`
- config_sha256: `12acd01fbad33cdc5feadf2fe54da0c7423960ab6f1bfa7c8a7005ff76b87e2f`
- source_seeds: `[0, 1, 2]`
- q_values: `[0.1, 0.5, 0.9]`
- squeue_absence: `True`
- stdout_validation_pass: `True`
- stderr_validation_pass: `True`
- verified_scheduler_handoff_count: `1`
- job_record_input_sha256: `82f8867157ddd1a459e852a42a71a96dfd7e2381938746997fb43537d7654122`
- job_record_launch_commit_matches: `True`
- job_record_runner_sha256_matches: `True`
- job_record_config_sha256_matches: `True`
- job_record_manifest_sha256_matches: `True`
- job_record_expected_units: `162`
- job_record_clean_launch: `True`
- job_record_max_concurrency: `8`
- job_record_monitoring_policy: `squeue_only_plus_artifact_validation`
- excluded_attempt_count: `6`
- recorded_excluded_attempt_count: `6`
- excluded_artifact_file_count: `18`
- all_excluded_attempts_verified: `True`
- all_excluded_attempts_have_zero_accepted_rows: `True`
- canceled_zero_result_job_ids: `['894872', '894902']`
- canceled_zero_result_jobs_contributed_no_units: `True`
- recorded_scheduler_handoff_count: `1`
- validation_pass: `True`
