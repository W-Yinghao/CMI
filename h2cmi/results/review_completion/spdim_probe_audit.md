# SPDIM Probe Audit

- status: PASS
- runner_commit: `54e855b18f765e6e6f043df146a261266383733e`
- official_repo: `https://github.com/fightlesliefigt/SPDIM`
- official_sha: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- external_path: `/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- external_license_file_present: `False`
- dataset: `BNCI2014_001`
- protocol: `W1-style LOSO, target first-session contiguous_split`
- subjects: `1,9`
- seed: `0`
- source_epochs: `20`
- adapt_epochs: `30`
- device: `cuda`
- elapsed_seconds: `133.396`

## Command

```bash
/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/run_spdim_probe.py --external-spdim-path /home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39 --dataset BNCI2014_001 --subjects 1,9 --seed 0 --epochs 20 --adapt-epochs 30 --device cuda --out h2cmi/results/review_completion/spdim_probe_results.csv --audit h2cmi/results/review_completion/spdim_probe_audit.md --failure-trace h2cmi/results/review_completion/spdim_probe_failure_trace.txt
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
| 9 | source_only | ok | 0.8333333333333333 | 0.8333333333333334 | 0.8300550747442959 |  |
| 9 | rct_refit | ok | 0.8055555555555556 | 0.8055555555555556 | 0.804953560371517 |  |
| 9 | spdim_geodesic | ok | 0.8055555555555556 | 0.8055555555555556 | 0.8054054054054054 |  |
| 9 | spdim_bias | ok | 0.8194444444444444 | 0.8194444444444444 | 0.8194096083349411 |  |

## Gate

- ok_rows: `8`
- failed_rows: `0`
- expected_ok_rows_for_success: `8`
