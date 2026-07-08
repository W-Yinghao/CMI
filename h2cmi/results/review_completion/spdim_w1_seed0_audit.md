# SPDIM W1 Seed-0 Audit

- status: PASS
- label: W1 seed-0 same-split official SPDIM expansion, not full three-seed baseline.
- monitoring policy: squeue only; sacct not used.
- result construction: monolithic partial rows plus clean non-overlapping shard rows.

## Row Gate

- expected_rows_total: `460`
- result_rows: `460`
- ok_rows: `460`
- failed_rows: `0`
- unique_keys: `460`
- duplicate_keys: `0`
- result_csv_sha256: `87ba93cac505e8d1d073bef67f29a4ccdd055e73185d637244ce2a3687c51698`

## Dataset Rows

| dataset | expected | rows | ok |
|---|---:|---:|---:|
| BNCI2014_001 | 36 | 36 | 36 |
| Cho2017 | 208 | 208 | 208 |
| Lee2019_MI | 216 | 216 | 216 |

## Slurm Jobs

| job | role | rows | final squeue | stderr status |
|---|---|---:|---|---|
| 889522 | monolithic_partial | 108 | absent | known_harmless_warnings_only |
| 889841 | cancelled_bad_wrapper_cho19_29 | 0 | absent | excluded_before_csv |
| 889842 | cancelled_bad_wrapper_cho30_40 | 0 | absent | excluded_before_csv |
| 889843 | cancelled_bad_wrapper_cho41_52 | 0 | absent | excluded_before_csv |
| 889844 | cancelled_bad_wrapper_lee01_11 | 0 | absent | excluded_before_csv |
| 889845 | cancelled_bad_wrapper_lee12_22 | 0 | absent | excluded_before_csv |
| 889846 | cancelled_bad_wrapper_lee23_33 | 0 | absent | excluded_before_csv |
| 889847 | cancelled_bad_wrapper_lee34_44 | 0 | absent | excluded_before_csv |
| 889848 | cancelled_bad_wrapper_lee45_54 | 0 | absent | excluded_before_csv |
| 889849 | shard_cho19_29 | 44 | absent | known_harmless_warnings_only |
| 889850 | shard_cho30_40 | 44 | absent | known_harmless_warnings_only |
| 889851 | shard_cho41_52 | 48 | absent | known_harmless_warnings_only |
| 889852 | shard_lee01_11 | 44 | absent | empty |
| 889853 | shard_lee12_22 | 44 | absent | empty |
| 889854 | shard_lee23_33 | 44 | absent | empty |
| 889855 | shard_lee34_44 | 44 | absent | empty |
| 889856 | shard_lee45_54 | 40 | absent | known_harmless_warnings_only |

## Clean Provenance Gate

- all result-carrying jobs launched from pushed commit `6a6e5b7758fe3d5130f87ea274be32ba6598dcd7`.
- all corrected shard stdout headers show an empty `repo_status_porcelain` block.
- all corrected shard stdout headers show external SPDIM SHA `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`.
- runner_dirty_allowed was not used.
- no official pretrained weights were used.
- no third-party SPDIM code was vendored.
- target labels were absent from adaptation loaders; P6A dry-run verified dummy adaptation labels and CPU forward passes for every dataset shape.

## Stderr Policy

- Cho2017 stderr contains only MOABB zero-buffer warnings from preprocessing.
- Lee2019_MI stderr is empty except `lee45_54`, which contains only the Matplotlib font-cache warning.
- The first bad wrapper jobs wrote a bad external-SHA echo to stderr and were cancelled before any CSV rows existed; they are explicitly excluded from the result.

## Red Team Review

- Scope check: only BNCI2014_001, Cho2017, Lee2019_MI; seed 0; four approved methods.
- No seeds 1/2, no full three-seed baseline, no TeX edits, no geometry stress, no orthogonal-score implementation.
- Completion check: final job state is absent from `squeue` plus artifact parse/count/checksum validation.
- Overclaim check: artifact is labeled W1 seed-0 same-split official SPDIM expansion, not a full three-seed baseline.
- Caveat check: Cho2017 single-class contiguous evaluation blocks are disclosed.
