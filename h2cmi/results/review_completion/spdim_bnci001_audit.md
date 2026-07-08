# Bounded BNCI2014-001 SPDIM Audit

- status: PASS
- runner_commit: `a749ba953b7f625cf713ab6673a569264c38af6a`
- official_repo: `https://github.com/fightlesliefigt/SPDIM`
- official_sha: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- external_path: `/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- external_license_file_present: `False`
- dataset: `BNCI2014_001`
- protocol: `W1-style LOSO, target first-session contiguous_split`
- subjects: `all`
- seed: `0`
- source_epochs: `20`
- adapt_epochs: `30`
- device: `cuda`
- elapsed_seconds: `486.583`

## Command

```bash
/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/run_spdim_probe.py --external-spdim-path /home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39 --dataset BNCI2014_001 --subjects all --seed 0 --epochs 20 --adapt-epochs 30 --device cuda --result-schema bnci001 --overwrite --allow-dirty --out h2cmi/results/review_completion/spdim_bnci001_results.csv --audit h2cmi/results/review_completion/spdim_bnci001_audit.md --failure-trace h2cmi/results/review_completion/spdim_bnci001_failure_trace.txt
```

## Target Label Policy

Target subject IDs are selected from sorted subject metadata only. Target adaptation datasets use dummy labels; target labels are read only by the evaluation metric code after adaptation/refit has completed.

## Results

| target | mode | status | bAcc | acc | macro-F1 | failure |
|---:|---|---|---:|---:|---:|---|
| 1 | source_only | ok | 0.5 | 0.5 | 0.3333333333333333 |  |
| 1 | rct_refit | ok | 0.8055555555555556 | 0.8055555555555556 | 0.8017309205350118 |  |
| 1 | spdim_geodesic | ok | 0.8055555555555556 | 0.8055555555555556 | 0.8017309205350118 |  |
| 1 | spdim_bias | ok | 0.8055555555555556 | 0.8055555555555556 | 0.8017309205350118 |  |
| 2 | source_only | ok | 0.5416666666666667 | 0.5416666666666666 | 0.4990512333965844 |  |
| 2 | rct_refit | ok | 0.5416666666666667 | 0.5416666666666666 | 0.5408695652173914 |  |
| 2 | spdim_geodesic | ok | 0.5138888888888888 | 0.5138888888888888 | 0.5137950993633031 |  |
| 2 | spdim_bias | ok | 0.513888888888889 | 0.5138888888888888 | 0.5092502434274586 |  |
| 3 | source_only | ok | 0.8333333333333333 | 0.8333333333333334 | 0.8300550747442959 |  |
| 3 | rct_refit | ok | 0.875 | 0.875 | 0.873807205452775 |  |
| 3 | spdim_geodesic | ok | 0.875 | 0.875 | 0.873807205452775 |  |
| 3 | spdim_bias | ok | 0.875 | 0.875 | 0.873807205452775 |  |
| 4 | source_only | ok | 0.5 | 0.5 | 0.3333333333333333 |  |
| 4 | rct_refit | ok | 0.7361111111111112 | 0.7361111111111112 | 0.7360601967972217 |  |
| 4 | spdim_geodesic | ok | 0.7222222222222222 | 0.7222222222222222 | 0.722007722007722 |  |
| 4 | spdim_bias | ok | 0.7222222222222222 | 0.7222222222222222 | 0.722007722007722 |  |
| 5 | source_only | ok | 0.5 | 0.5 | 0.3333333333333333 |  |
| 5 | rct_refit | ok | 0.5972222222222222 | 0.5972222222222222 | 0.5965217391304347 |  |
| 5 | spdim_geodesic | ok | 0.5694444444444444 | 0.5694444444444444 | 0.562610229276896 |  |
| 5 | spdim_bias | ok | 0.5555555555555556 | 0.5555555555555556 | 0.5428571428571429 |  |
| 6 | source_only | ok | 0.5277777777777778 | 0.5277777777777778 | 0.39225422045680236 |  |
| 6 | rct_refit | ok | 0.75 | 0.75 | 0.746875 |  |
| 6 | spdim_geodesic | ok | 0.75 | 0.75 | 0.746875 |  |
| 6 | spdim_bias | ok | 0.7361111111111112 | 0.7361111111111112 | 0.7335929892891919 |  |
| 7 | source_only | ok | 0.6805555555555556 | 0.6805555555555556 | 0.6799999999999999 |  |
| 7 | rct_refit | ok | 0.6388888888888888 | 0.6388888888888888 | 0.6377708978328174 |  |
| 7 | spdim_geodesic | ok | 0.6388888888888888 | 0.6388888888888888 | 0.6377708978328174 |  |
| 7 | spdim_bias | ok | 0.6527777777777777 | 0.6527777777777778 | 0.6527107852595022 |  |
| 8 | source_only | ok | 0.7222222222222222 | 0.7222222222222222 | 0.7037037037037037 |  |
| 8 | rct_refit | ok | 0.8194444444444444 | 0.8194444444444444 | 0.8194096083349411 |  |
| 8 | spdim_geodesic | ok | 0.8194444444444444 | 0.8194444444444444 | 0.8194096083349411 |  |
| 8 | spdim_bias | ok | 0.8194444444444444 | 0.8194444444444444 | 0.8194096083349411 |  |
| 9 | source_only | ok | 0.8333333333333333 | 0.8333333333333334 | 0.8300550747442959 |  |
| 9 | rct_refit | ok | 0.8055555555555556 | 0.8055555555555556 | 0.804953560371517 |  |
| 9 | spdim_geodesic | ok | 0.8194444444444444 | 0.8194444444444444 | 0.8191304347826087 |  |
| 9 | spdim_bias | ok | 0.8194444444444444 | 0.8194444444444444 | 0.8194096083349411 |  |

## Gate

- ok_rows: `36`
- failed_rows: `0`
- expected_ok_rows_for_success: `36`

## Slurm Completion Validation

- job_id: `888854`
- submit_command: `sbatch h2cmi/results/review_completion/slurm/spdim_bnci001.slurm`
- monitor_policy: `squeue only; sacct not used`
- final_squeue_absence: `true`
- final_squeue_command: `squeue -j 888854 -o '%i %T %M %D %R'`
- final_squeue_result: header only, no job row
- stderr_path: `h2cmi/results/review_completion/slurm/logs/spdim-bnci001-888854.err`
- stderr_status: `empty`
- stdout_path: `h2cmi/results/review_completion/slurm/logs/spdim-bnci001-888854.out`
- stdout_status: `exists, 27556 bytes`
- stdout_sha256: `e94d972082eae5095c27e7a6014287bd1b6b67705b4a42972abbdde6ae7a5068`
- stderr_sha256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- failure_trace_status: `absent`

## Artifact Validation

- results_path: `h2cmi/results/review_completion/spdim_bnci001_results.csv`
- results_sha256: `b0ccaaa05c00ca9209224a728d39bbdc71b17c7989c28673257fb89886e43a7e`
- audit_sha256_before_validation_append: `bf8befc6db81271327f022ee9be12d3bb2b3d5640c5de7aee9027602f7a14b70`
- csv_parse: `pass`
- expected_rows: `36`
- observed_rows: `36`
- status_ok_rows: `36`
- failed_rows: `0`
- targets_observed: `1 2 3 4 5 6 7 8 9`
- methods_observed: `source_only_tsmnet rct spdim_geodesic spdim_bias`
- n_eval_per_row: `72`
- class_counts_eval_per_row: `[36,36]`
- prediction_hash_status: `64-hex present for every row`
- logits_hash_status: `64-hex present for every row`
- unique_prediction_hashes_by_method: `source_only_tsmnet=8, rct=9, spdim_geodesic=9, spdim_bias=9`
- unique_logits_hashes_by_method: `source_only_tsmnet=9, rct=9, spdim_geodesic=9, spdim_bias=9`

## Integrity Gates

- P4.1 probe_integrity_audit: `PASS`
- target_label_leakage_detected: `false`
- fallback_prediction_detected: `false`
- split_mismatch_detected: `false`
- pretrained_weight_detected: `false`
- third_party_vendored: `false`
- official_pretrained_weights_used: `false`
- official_spdim_sha: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- runner_commit_at_launch: `a749ba953b7f625cf713ab6673a569264c38af6a`
- runner_dirty_allowed: `true`
- runner_dirty_reason: `PM requested one commit after P4.1 and P5; the P5 runner/protocol changes were intentionally uncommitted during the Slurm run and are recorded by stdout.`
- runner_diff_sha256_at_launch: `870ca4e40c417a0fbd80ee63e9833e3cc22bb727388fa350f7eb21d748e9ca82`

Verdict: PASS for the bounded BNCI2014-001 expansion gate. This is not a full
W1 SPDIM sweep and does not include Cho2017, Lee2019-MI, additional seeds, or
target-label-based subject filtering.
