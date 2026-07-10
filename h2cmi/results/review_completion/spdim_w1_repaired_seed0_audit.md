# SPDIM W1 Repaired-Split Seed-0 Audit

- status: `pass`
- label: W1 repaired-split seed-0 official SPDIM expansion, not full three-seed baseline.
- monitoring policy: `squeue` plus artifact-level completion gates
- result construction: four non-overlapping clean shards; canceled monolithic partial excluded

## Row Gate

- expected_rows_total: `460`
- result_rows: `460`
- status_rows: `{"ok": 460}`
- duplicate_keys: `0`
- result_csv_sha256: `118ec37f3a195d50c24abf24b4c61048cdbc0ffff7d9c0f0bf51c83f7f69229c`
- prediction_hash_missing_rows: `0`
- logits_hash_missing_rows: `0`

## Dataset Rows

| dataset | subjects | expected rows | actual rows |
|---|---:|---:|---:|
| BNCI2014_001 | 9 | 36 | 36 |
| Cho2017 | 52 | 208 | 208 |
| Lee2019_MI | 54 | 216 | 216 |

## Slurm and Shard Evidence

| shard | job id | target spec | rows | final squeue | stdout | stderr | result sha256 |
|---|---:|---|---:|---|---|---|---|
| shard0 | 891457 | `BNCI2014_001=1-9;Cho2017=1-20` | 116 | absent | exists_nonempty_clean_launch_header | known_harmless_warnings_only | `6837ac032d66623daf0fdafccc58530db3ac5798c5111a1d9b90e4bd7774b87c` |
| shard1 | 891458 | `Cho2017=21-49` | 116 | absent | exists_nonempty_clean_launch_header | known_harmless_warnings_only | `2e3c81ed3e98c5fd4d8949eb9d325635c75b48e66a12ea7f7aff7813a303e1da` |
| shard2 | 891459 | `Cho2017=50-52;Lee2019_MI=1-26` | 116 | absent | exists_nonempty_clean_launch_header | known_harmless_warnings_only | `6249945538afae247be4b00f0df0e86e867272852d108dd8625f4741f37d21ea` |
| shard3 | 891456 | `Lee2019_MI=27-54` | 112 | absent | exists_nonempty_clean_launch_header | empty | `297fc0619187edf96396685906becdd5a4323db5ad6a31b845bd828dbf9cdbd1` |

Accepted completion rule: job absent from `squeue` and shard/final CSV parse, row-count, status, provenance, and checksum gates all passed. Stdout is present with the clean launch header. Stderr is empty or contains only the declared MOABB zero-buffer warning and one Matplotlib font-cache notice.

The canceled monolithic job `891435` wrote 56 partial rows with SHA-256 `ed434d8927b2ee73d6839ca1ba0c724de31118e2ed1e5a8230dc362adae341dc`; those rows are preserved outside the repository and excluded from every result statistic.

## Clean Provenance Gate

- launch_commit: `763e11c4412938017f0a7b1be3cfbe9e40ec3d41`
- repaired_manifest_hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- runner_sha256: `946b28b93f0ddbce395ade7c6a13d30b20f368fe7a1ae22fbefa01f291e82be8`
- config_sha256: `6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b`
- external_spdim_commit: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- clean_worktree_at_launch: `true` for all four shards
- runner_dirty_allowed: `false` for all four shards
- result commit preserves the launched runner, config, and manifest file checksums
- launcher_python: `/home/infres/yinwang/anaconda3/envs/icml/bin/python`
- recorded_environment_name: `base`
- environment note: the runner recorded inherited `CONDA_DEFAULT_ENV=base`; the pinned Slurm launcher invoked the `icml` interpreter path above, and the launcher matches the launch commit byte-for-byte

## Split and Leakage Gate

- single_class_eval_rows: `0`
- single_class_adapt_rows: `0`
- adapt_eval_disjoint_failures: `0`
- target_label_leakage_detected: `False`
- target_performance_method_selection_detected: `False`
- pretrained_weight_detected: `False`
- vendoring_detected: `False`

## Red Team Review

- Raw shard CSVs, summaries, and normalized text logs are committed with the final packet; the merge is reproducible from those inputs.
- Subject/method keys are exact and non-overlapping across shards; 115 subjects times four methods yields 460 rows.
- The repaired split removes the old Cho2017 single-class evaluation defect, but this run remains seed 0 only.
- The recorded environment name is an inherited shell label; interpreter provenance comes from the pinned launcher path and launch-commit match.
- No seeds 1/2, full SPDIM, H2CMI rerun, TeX edit, geometry stress, or orthogonal-score work is included.
- No inferential CI or full-baseline claim is made.
