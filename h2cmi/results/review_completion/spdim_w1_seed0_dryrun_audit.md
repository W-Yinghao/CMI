# SPDIM W1 Seed-0 Dry-Run Audit

- status: PASS
- launch_commit: `68414a9a349d4383f06a5f4288a8668b29d91444`
- external_spdim_commit: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- official_repo: `https://github.com/fightlesliefigt/SPDIM`
- expected_rows_total: `460`
- approve_gpu_run: `True`
- estimated_gpu_hours: `18.0`

## Scope

- datasets: `BNCI2014_001`, `Cho2017`, `Lee2019_MI` (`Lee2019-MI` in PM wording).
- split: exact H2CMI W1-style LOSO, target first-session contiguous adaptation/evaluation blocks.
- source seed: `0` only.
- methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.
- no full three-seed run.

## Gate Checks

- target_label_leakage_detected: `False`
- pretrained_weight_detected: `False`
- vendoring_detected: `False`
- shape_blocker_detected: `False`
- official SPDIM/TSMNet model instantiation: passed for every dataset.
- one CPU forward pass without target labels: passed for every dataset.
- adaptation loader dummy labels: all checked loaders contained only dummy label `0`.
- evaluation labels are used for split audit/evaluation counts only after adaptation-loader construction.

## Dataset Summary

| dataset | targets | tensor shape | channel count | expected rows | eval class-count range | dry-run seconds |
|---|---:|---|---:|---:|---|---:|
| BNCI2014_001 | 9 | `[2592, 22, 500]` | 22 | 36 | 36-36 / 36-36 | 30.3 |
| Cho2017 | 52 | `[10520, 64, 500]` | 64 | 208 | 0-0 / 100-120 | 476.4 |
| Lee2019_MI | 54 | `[10800, 62, 500]` | 62 | 216 | 21-30 / 20-29 | 2178.5 |

## Split Evidence

Exact per-target source subject IDs, adaptation indices, evaluation indices, split SHA-256 values, and evaluation class counts are in the machine-readable JSON audit.

## Blockers

- none
