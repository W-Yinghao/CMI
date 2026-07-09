# W1 Valid-Subset Recompute

- status: PASS
- label: `legacy_valid_subset_diagnostic_only`
- not_confirmatory_full_W1: `true`
- included datasets: `BNCI2014_001`, `Lee2019_MI`
- excluded dataset: `Cho2017`
- source artifacts only; no model rerun.

## H2CMI W1 Four-Branch bAcc

| branch | subject-weighted n | subject-weighted mean | dataset-macro mean | BNCI2014_001 | Lee2019_MI |
|---|---:|---:|---:|---:|---:|
| identity_uniform | 63 | 0.709625 | 0.701156 | 0.689300 | 0.713012 |
| identity_joint_prior | 63 | 0.699274 | 0.691475 | 0.680556 | 0.702393 |
| joint_geometry_uniform | 63 | 0.718600 | 0.711750 | 0.702160 | 0.721340 |
| joint_geometry_joint_prior | 63 | 0.714960 | 0.708769 | 0.700103 | 0.717436 |

## H2CMI W1 Decomposition And Contrasts

| metric | subject-weighted mean | dataset-macro mean |
|---|---:|---:|
| G | 0.008975 | 0.010594 |
| P | -0.010351 | -0.009682 |
| interaction | 0.006711 | 0.006701 |
| full_joint_delta | 0.005335 | 0.007613 |
| prior_m_step_geometry | 0.004572 | 0.007382 |
| fixed_iterative_minus_joint_geometry | 0.004572 | 0.007382 |
| joint_geometry_minus_pooled | -0.005526 | -0.010511 |

## Cho2017 Dependence

- full all-dataset G subject-weighted mean: `0.060419`
- Cho2017-only G subject-weighted mean: `0.122746`
- valid-subset G subject-weighted mean: `0.008975`
- valid-subset fixed-prior iterative minus joint subject-weighted mean: `0.004572`
- verdict: previous MI geometry aggregate magnitude depends on Cho2017; valid-subset results are diagnostic only.

## SPDIM Seed-0 Valid Subset

| method | subject-weighted mean bAcc | dataset-macro mean bAcc | BNCI2014_001 | Lee2019_MI |
|---|---:|---:|---:|---:|
| source_only_tsmnet | 0.561198 | 0.587782 | 0.625000 | 0.550565 |
| rct | 0.686865 | 0.704812 | 0.729938 | 0.679687 |
| spdim_geodesic | 0.686655 | 0.701474 | 0.722222 | 0.680727 |
| spdim_bias | 0.682021 | 0.698772 | 0.722222 | 0.675321 |

## SPDIM Deltas

| contrast | subject-weighted mean bAcc delta | dataset-macro mean bAcc delta |
|---|---:|---:|
| rct_minus_source_only_tsmnet | 0.125667 | 0.117030 |
| spdim_geodesic_minus_source_only_tsmnet | 0.125456 | 0.113692 |
| spdim_bias_minus_source_only_tsmnet | 0.120823 | 0.110989 |
| spdim_geodesic_minus_rct | -0.000211 | -0.003338 |
| spdim_bias_minus_rct | -0.004844 | -0.006041 |

## Red Team Review

- Cho2017 is excluded from every valid-subset row.
- All rows are labeled diagnostic-only and not confirmatory full W1.
- Existing raw/metric artifacts only; no model rerun.
