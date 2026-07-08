# SPDIM Probe Audit

- status: PASS
- runner_commit: `a8b93682c152a428f9689f9941efaff486606336`
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
- elapsed_seconds: `800.558`

## Command

```bash
/home/infres/yinwang/CMI_AAAI_spdim_clean_a8b9368/h2cmi/run_spdim_probe.py --external-spdim-path /home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39 --dataset BNCI2014_001 --subjects all --seed 0 --epochs 20 --adapt-epochs 30 --device cuda --result-schema bnci001 --overwrite --out h2cmi/results/review_completion/spdim_bnci001_clean_results.csv --audit h2cmi/results/review_completion/spdim_bnci001_clean_audit.md --failure-trace h2cmi/results/review_completion/spdim_bnci001_clean_failure_trace.txt
```

## Launch Provenance

- launch_commit: `a8b93682c152a428f9689f9941efaff486606336`
- clean_worktree: `True`
- runner_dirty_allowed: `False`
- runner_file: `h2cmi/run_spdim_probe.py`
- runner_file_sha256: `5ccdccdcdcfacd6cc27a335ee44afa06842b469691ac53cb4e4ef0930e760490`
- config_file: `h2cmi/config.py`
- config_sha256: `6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b`
- external_spdim_commit: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- environment_name: `base`
- slurm_job_id: `889192`

### Git Status Porcelain

```text
(empty)
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

## P5.2 Clean Validation

- status: PASS
- Slurm job id: `889192`
- Slurm monitoring: `squeue` only; `sacct` was not used.
- Final `squeue -j 889192 -o '%i %T %M %D %R'`: job absent from queue.
- stderr status: empty (`h2cmi/results/review_completion/slurm/logs/spdim-bnci001-clean-889192.err`, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
- stdout status: exists (`h2cmi/results/review_completion/slurm/logs/spdim-bnci001-clean-889192.out`, SHA-256 `2ee209d80aadb7c02800ed3b203d98ffd4f925309a13a3b28e281584b087c03c`).
- result CSV status: parsed, 36 data rows, 36 `ok`, SHA-256 `4b8e17542220511baddb41bdfc412dde68b38a214e813affdd7348c99d4d6338`.
- summary JSON status: parsed.
- comparison CSV status: parsed, 36 data rows.
- target-label leakage detected: no.
- target-label-based filtering or model selection detected: no.
- official pretrained weights detected: no.
- third-party vendoring detected: no.

## Clean Provenance Gate

- launch commit equals pushed guard commit: yes (`a8b93682c152a428f9689f9941efaff486606336`).
- clean worktree at launch: yes; launch stdout shows an empty `repo_status_porcelain` block and the audit records `(empty)`.
- `runner_dirty_allowed`: `False`.
- runner checksum at launch: `5ccdccdcdcfacd6cc27a335ee44afa06842b469691ac53cb4e4ef0930e760490`.
- config checksum at launch: `6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b`.
- external SPDIM commit: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`.
- post-launch runner/config diff before result commit: empty.
- result commit contains exactly the launched runner/config state: yes; P5.2B stages only review-completion result artifacts and command log updates, not runner/config changes.

## Comparison To Exploratory P5

- exploratory file: `h2cmi/results/review_completion/spdim_bnci001_results.csv`.
- clean file: `h2cmi/results/review_completion/spdim_bnci001_clean_results.csv`.
- acc mismatches: `0/36`.
- bAcc mismatches: `0/36`.
- prediction-hash mismatches: `0/36`.
- logits-hash mismatches: `36/36`.

The clean rerun reproduces the exploratory run at the metric and discrete-prediction level for every subject/method row. The continuous logits are not byte-identical across the dirty exploratory run and clean rerun, so the logits mismatch is recorded as a real mismatch rather than hidden. This does not indicate a fallback/default prediction table: every row has a real prediction hash, nonempty logits hash, parsed evaluation count, and the same final predictions as the exploratory run.
