# Official SPDIM W1 Repaired-Split Three-Seed Protocol

Pre-registered final label, allowed only if all P9 gates pass: Official SPDIM W1 repaired-split three-source-seed same-split baseline.

## Frozen Components

- P8 result commit: `3d820dfd1ef988cdd44acd34d47ed37c490a98e5`.
- repaired split manifest hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`.
- P8 runner SHA-256: `946b28b93f0ddbce395ade7c6a13d30b20f368fe7a1ae22fbefa01f291e82be8`.
- config SHA-256: `6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b`.
- external SPDIM commit: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`.
- immutable P8 seed-0 result SHA-256: `118ec37f3a195d50c24abf24b4c61048cdbc0ffff7d9c0f0bf51c83f7f69229c`.
- source-training and adaptation hyperparameters are unchanged from P8.

## Scope

- new source seeds: `1`, `2` only.
- datasets: `BNCI2014_001`, `Cho2017`, `Lee2019_MI`.
- target subjects: all 115 W1 repaired-split targets per seed.
- methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.
- epochs: `20`; adaptation epochs: `30`; adaptation LR: `0.01`.
- batch size: `64`; source validation fraction: `0.2`.
- temporal filters: `4`; spatial filters: `40`; subspace dimensions: `20`.
- model dtype: `float32`; SPD calculation device: `cpu`.
- no target-performance tuning or method selection.
- no official pretrained weights and no third-party vendoring.

## Expected Rows

| dataset | targets per seed | seeds | methods | new rows | final rows including seed 0 |
|---|---:|---:|---:|---:|---:|
| BNCI2014_001 | 9 | 2 | 4 | 72 | 108 |
| Cho2017 | 52 | 2 | 4 | 416 | 624 |
| Lee2019_MI | 54 | 2 | 4 | 432 | 648 |
| total | 115 | 2 | 4 | 920 | 1380 |

## GPU Sharding

Eight immutable seed-by-target-shard tasks are submitted as one array with concurrency `%4`, preferring `H100,L40S`. Each seed reuses the four P8 target shards with expected rows `116`, `116`, `116`, and `112`. A failed task may be rerun only with the same seed, target spec, and frozen command.

## Aggregation and Inference

- Average seeds 0/1/2 within each dataset x target subject x method before aggregation.
- Primary estimands: subject-weighted and dataset-macro means.
- Cluster bootstrap: `10000` replicates, fixed seed `20260710`, cluster unit dataset x target subject.
- Preserve all methods and contrasts within each sampled subject; report percentile 95% CIs.
- Harm thresholds: delta `< 0`, `< -0.01`, and `< -0.02`.
- No post-hoc equivalence or noninferiority margin.

## Completion and Failure Gates

- Monitor with `squeue`; completion requires queue absence plus stdout/stderr and artifact validation.
- Require 920 new rows and a deterministic 1380-row merge preserving seed 0 byte-for-byte.
- Require both classes, disjoint adaptation/evaluation IDs, complete prediction/logits hashes, frozen checksums, and no leakage flags.
- Stop and write a failure trace on any unresolved row, provenance, checksum, or scope failure.

## Runtime Provenance Policy

Every GPU task records `sys.executable`, `sys.prefix`, Python, PyTorch, CUDA, GPU device, MOABB, and MNE versions. `CONDA_DEFAULT_ENV` is retained only as an inherited shell label and is not runtime proof.

## Red Team Review

- P8 runner, config, manifest, methods, and hyperparameters are frozen before seeds 1/2 are observed.
- Seed 0 provides no SPDIM-specific improvement over RCT; P9 does not alter the protocol in response.
- No H2CMI rerun, TeX edit, geometry stress, orthogonal-score work, extra seed, or extra method is authorized.
