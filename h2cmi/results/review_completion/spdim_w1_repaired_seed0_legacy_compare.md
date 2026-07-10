# SPDIM W1 Repaired Seed-0 Legacy Compare

The old P6 SPDIM result used the quarantined contiguous split and remains diagnostic only. The repaired result uses the frozen P7 `class_stratified_half` manifest. This comparison reports magnitude changes; it does not rehabilitate the old split.

## Per-Dataset Mean bAcc

| dataset | method | old diagnostic | repaired seed-0 | repaired - old |
|---|---|---:|---:|---:|
| BNCI2014_001 | source_only_tsmnet | 0.625000 | 0.626543 | +0.001543 |
| BNCI2014_001 | rct | 0.729938 | 0.729938 | +0.000000 |
| BNCI2014_001 | spdim_geodesic | 0.722222 | 0.725309 | +0.003086 |
| BNCI2014_001 | spdim_bias | 0.722222 | 0.722222 | +0.000000 |
| Cho2017 | source_only_tsmnet | 0.894199 | 0.528558 | -0.365641 |
| Cho2017 | rct | 0.684968 | 0.599776 | -0.085192 |
| Cho2017 | spdim_geodesic | 0.669391 | 0.599199 | -0.070192 |
| Cho2017 | spdim_bias | 0.670609 | 0.598429 | -0.072179 |
| Lee2019_MI | source_only_tsmnet | 0.550565 | 0.550000 | -0.000565 |
| Lee2019_MI | rct | 0.679687 | 0.682222 | +0.002536 |
| Lee2019_MI | spdim_geodesic | 0.680727 | 0.679259 | -0.001467 |
| Lee2019_MI | spdim_bias | 0.675321 | 0.671481 | -0.003839 |

## Aggregate Mean bAcc

| method | old subject-weighted | repaired subject-weighted | old dataset-macro | repaired dataset-macro |
|---|---:|---:|---:|---:|
| source_only_tsmnet | 0.711772 | 0.546295 | 0.689921 | 0.568367 |
| rct | 0.686007 | 0.648676 | 0.698198 | 0.670645 |
| spdim_geodesic | 0.678848 | 0.646662 | 0.690780 | 0.667922 |
| spdim_bias | 0.676861 | 0.642420 | 0.689384 | 0.664044 |

## Red Team Review

- The split change alters the estimand, especially for Cho2017; differences are not paired treatment effects.
- P6 remains legacy diagnostic only.
- The repaired result is seed 0 only and is not a full three-seed baseline.
