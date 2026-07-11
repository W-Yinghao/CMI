# Repaired-W1 Cross-Pipeline Harm Audit

Every delta is seed-averaged at the target-subject level and is relative to the method's own pipeline baseline.

Harm rates must not be compared as if H2CMI identity and source-only TSMNet were the same source model.

For dataset-macro rows, an integer harm count is not mathematically defined because the estimand is the arithmetic mean of three unequal-denominator dataset rates. The CSV reports `NA` for that count, denominator `3`, the macro rate, and raw subject counts only as context.

| pipeline | method | threshold | scope | dataset | harm count | denominator | harm rate |
|---|---|---:|---|---|---:|---:|---:|
| h2cmi | pooled | +0.00 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| h2cmi | pooled | +0.00 | per_dataset | Cho2017 | 17 | 52 | 0.3269 |
| h2cmi | pooled | +0.00 | per_dataset | Lee2019_MI | 13 | 54 | 0.2407 |
| h2cmi | pooled | +0.00 | subject_weighted | ALL | 31 | 115 | 0.2696 |
| h2cmi | pooled | +0.00 | dataset_macro | ALL | NA | 3 | 0.2263 |
| h2cmi | pooled | -0.01 | per_dataset | BNCI2014_001 | 0 | 9 | 0.0000 |
| h2cmi | pooled | -0.01 | per_dataset | Cho2017 | 13 | 52 | 0.2500 |
| h2cmi | pooled | -0.01 | per_dataset | Lee2019_MI | 12 | 54 | 0.2222 |
| h2cmi | pooled | -0.01 | subject_weighted | ALL | 25 | 115 | 0.2174 |
| h2cmi | pooled | -0.01 | dataset_macro | ALL | NA | 3 | 0.1574 |
| h2cmi | pooled | -0.02 | per_dataset | BNCI2014_001 | 0 | 9 | 0.0000 |
| h2cmi | pooled | -0.02 | per_dataset | Cho2017 | 6 | 52 | 0.1154 |
| h2cmi | pooled | -0.02 | per_dataset | Lee2019_MI | 7 | 54 | 0.1296 |
| h2cmi | pooled | -0.02 | subject_weighted | ALL | 13 | 115 | 0.1130 |
| h2cmi | pooled | -0.02 | dataset_macro | ALL | NA | 3 | 0.0817 |
| h2cmi | FRSC | +0.00 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| h2cmi | FRSC | +0.00 | per_dataset | Cho2017 | 18 | 52 | 0.3462 |
| h2cmi | FRSC | +0.00 | per_dataset | Lee2019_MI | 16 | 54 | 0.2963 |
| h2cmi | FRSC | +0.00 | subject_weighted | ALL | 35 | 115 | 0.3043 |
| h2cmi | FRSC | +0.00 | dataset_macro | ALL | NA | 3 | 0.2512 |
| h2cmi | FRSC | -0.01 | per_dataset | BNCI2014_001 | 0 | 9 | 0.0000 |
| h2cmi | FRSC | -0.01 | per_dataset | Cho2017 | 9 | 52 | 0.1731 |
| h2cmi | FRSC | -0.01 | per_dataset | Lee2019_MI | 12 | 54 | 0.2222 |
| h2cmi | FRSC | -0.01 | subject_weighted | ALL | 21 | 115 | 0.1826 |
| h2cmi | FRSC | -0.01 | dataset_macro | ALL | NA | 3 | 0.1318 |
| h2cmi | FRSC | -0.02 | per_dataset | BNCI2014_001 | 0 | 9 | 0.0000 |
| h2cmi | FRSC | -0.02 | per_dataset | Cho2017 | 4 | 52 | 0.0769 |
| h2cmi | FRSC | -0.02 | per_dataset | Lee2019_MI | 8 | 54 | 0.1481 |
| h2cmi | FRSC | -0.02 | subject_weighted | ALL | 12 | 115 | 0.1043 |
| h2cmi | FRSC | -0.02 | dataset_macro | ALL | NA | 3 | 0.0750 |
| h2cmi | fixed-prior iterative, uniform decision prior | +0.00 | per_dataset | BNCI2014_001 | 2 | 9 | 0.2222 |
| h2cmi | fixed-prior iterative, uniform decision prior | +0.00 | per_dataset | Cho2017 | 19 | 52 | 0.3654 |
| h2cmi | fixed-prior iterative, uniform decision prior | +0.00 | per_dataset | Lee2019_MI | 17 | 54 | 0.3148 |
| h2cmi | fixed-prior iterative, uniform decision prior | +0.00 | subject_weighted | ALL | 38 | 115 | 0.3304 |
| h2cmi | fixed-prior iterative, uniform decision prior | +0.00 | dataset_macro | ALL | NA | 3 | 0.3008 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.01 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.01 | per_dataset | Cho2017 | 13 | 52 | 0.2500 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.01 | per_dataset | Lee2019_MI | 11 | 54 | 0.2037 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.01 | subject_weighted | ALL | 25 | 115 | 0.2174 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.01 | dataset_macro | ALL | NA | 3 | 0.1883 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.02 | per_dataset | BNCI2014_001 | 0 | 9 | 0.0000 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.02 | per_dataset | Cho2017 | 5 | 52 | 0.0962 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.02 | per_dataset | Lee2019_MI | 7 | 54 | 0.1296 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.02 | subject_weighted | ALL | 12 | 115 | 0.1043 |
| h2cmi | fixed-prior iterative, uniform decision prior | -0.02 | dataset_macro | ALL | NA | 3 | 0.0753 |
| h2cmi | joint-fit geometry, uniform decision prior | +0.00 | per_dataset | BNCI2014_001 | 3 | 9 | 0.3333 |
| h2cmi | joint-fit geometry, uniform decision prior | +0.00 | per_dataset | Cho2017 | 21 | 52 | 0.4038 |
| h2cmi | joint-fit geometry, uniform decision prior | +0.00 | per_dataset | Lee2019_MI | 26 | 54 | 0.4815 |
| h2cmi | joint-fit geometry, uniform decision prior | +0.00 | subject_weighted | ALL | 50 | 115 | 0.4348 |
| h2cmi | joint-fit geometry, uniform decision prior | +0.00 | dataset_macro | ALL | NA | 3 | 0.4062 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.01 | per_dataset | BNCI2014_001 | 2 | 9 | 0.2222 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.01 | per_dataset | Cho2017 | 13 | 52 | 0.2500 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.01 | per_dataset | Lee2019_MI | 16 | 54 | 0.2963 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.01 | subject_weighted | ALL | 31 | 115 | 0.2696 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.01 | dataset_macro | ALL | NA | 3 | 0.2562 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.02 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.02 | per_dataset | Cho2017 | 5 | 52 | 0.0962 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.02 | per_dataset | Lee2019_MI | 10 | 54 | 0.1852 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.02 | subject_weighted | ALL | 16 | 115 | 0.1391 |
| h2cmi | joint-fit geometry, uniform decision prior | -0.02 | dataset_macro | ALL | NA | 3 | 0.1308 |
| h2cmi | Latent-IM-Diag | +0.00 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| h2cmi | Latent-IM-Diag | +0.00 | per_dataset | Cho2017 | 17 | 52 | 0.3269 |
| h2cmi | Latent-IM-Diag | +0.00 | per_dataset | Lee2019_MI | 20 | 54 | 0.3704 |
| h2cmi | Latent-IM-Diag | +0.00 | subject_weighted | ALL | 38 | 115 | 0.3304 |
| h2cmi | Latent-IM-Diag | +0.00 | dataset_macro | ALL | NA | 3 | 0.2695 |
| h2cmi | Latent-IM-Diag | -0.01 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| h2cmi | Latent-IM-Diag | -0.01 | per_dataset | Cho2017 | 10 | 52 | 0.1923 |
| h2cmi | Latent-IM-Diag | -0.01 | per_dataset | Lee2019_MI | 16 | 54 | 0.2963 |
| h2cmi | Latent-IM-Diag | -0.01 | subject_weighted | ALL | 27 | 115 | 0.2348 |
| h2cmi | Latent-IM-Diag | -0.01 | dataset_macro | ALL | NA | 3 | 0.1999 |
| h2cmi | Latent-IM-Diag | -0.02 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| h2cmi | Latent-IM-Diag | -0.02 | per_dataset | Cho2017 | 5 | 52 | 0.0962 |
| h2cmi | Latent-IM-Diag | -0.02 | per_dataset | Lee2019_MI | 11 | 54 | 0.2037 |
| h2cmi | Latent-IM-Diag | -0.02 | subject_weighted | ALL | 17 | 115 | 0.1478 |
| h2cmi | Latent-IM-Diag | -0.02 | dataset_macro | ALL | NA | 3 | 0.1370 |
| spdim | RCT | +0.00 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | RCT | +0.00 | per_dataset | Cho2017 | 2 | 52 | 0.0385 |
| spdim | RCT | +0.00 | per_dataset | Lee2019_MI | 3 | 54 | 0.0556 |
| spdim | RCT | +0.00 | subject_weighted | ALL | 6 | 115 | 0.0522 |
| spdim | RCT | +0.00 | dataset_macro | ALL | NA | 3 | 0.0684 |
| spdim | RCT | -0.01 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | RCT | -0.01 | per_dataset | Cho2017 | 1 | 52 | 0.0192 |
| spdim | RCT | -0.01 | per_dataset | Lee2019_MI | 1 | 54 | 0.0185 |
| spdim | RCT | -0.01 | subject_weighted | ALL | 3 | 115 | 0.0261 |
| spdim | RCT | -0.01 | dataset_macro | ALL | NA | 3 | 0.0496 |
| spdim | RCT | -0.02 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | RCT | -0.02 | per_dataset | Cho2017 | 0 | 52 | 0.0000 |
| spdim | RCT | -0.02 | per_dataset | Lee2019_MI | 1 | 54 | 0.0185 |
| spdim | RCT | -0.02 | subject_weighted | ALL | 2 | 115 | 0.0174 |
| spdim | RCT | -0.02 | dataset_macro | ALL | NA | 3 | 0.0432 |
| spdim | SPDIM geodesic | +0.00 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | SPDIM geodesic | +0.00 | per_dataset | Cho2017 | 2 | 52 | 0.0385 |
| spdim | SPDIM geodesic | +0.00 | per_dataset | Lee2019_MI | 3 | 54 | 0.0556 |
| spdim | SPDIM geodesic | +0.00 | subject_weighted | ALL | 6 | 115 | 0.0522 |
| spdim | SPDIM geodesic | +0.00 | dataset_macro | ALL | NA | 3 | 0.0684 |
| spdim | SPDIM geodesic | -0.01 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | SPDIM geodesic | -0.01 | per_dataset | Cho2017 | 1 | 52 | 0.0192 |
| spdim | SPDIM geodesic | -0.01 | per_dataset | Lee2019_MI | 2 | 54 | 0.0370 |
| spdim | SPDIM geodesic | -0.01 | subject_weighted | ALL | 4 | 115 | 0.0348 |
| spdim | SPDIM geodesic | -0.01 | dataset_macro | ALL | NA | 3 | 0.0558 |
| spdim | SPDIM geodesic | -0.02 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | SPDIM geodesic | -0.02 | per_dataset | Cho2017 | 0 | 52 | 0.0000 |
| spdim | SPDIM geodesic | -0.02 | per_dataset | Lee2019_MI | 2 | 54 | 0.0370 |
| spdim | SPDIM geodesic | -0.02 | subject_weighted | ALL | 3 | 115 | 0.0261 |
| spdim | SPDIM geodesic | -0.02 | dataset_macro | ALL | NA | 3 | 0.0494 |
| spdim | SPDIM bias | +0.00 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | SPDIM bias | +0.00 | per_dataset | Cho2017 | 2 | 52 | 0.0385 |
| spdim | SPDIM bias | +0.00 | per_dataset | Lee2019_MI | 3 | 54 | 0.0556 |
| spdim | SPDIM bias | +0.00 | subject_weighted | ALL | 6 | 115 | 0.0522 |
| spdim | SPDIM bias | +0.00 | dataset_macro | ALL | NA | 3 | 0.0684 |
| spdim | SPDIM bias | -0.01 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | SPDIM bias | -0.01 | per_dataset | Cho2017 | 1 | 52 | 0.0192 |
| spdim | SPDIM bias | -0.01 | per_dataset | Lee2019_MI | 3 | 54 | 0.0556 |
| spdim | SPDIM bias | -0.01 | subject_weighted | ALL | 5 | 115 | 0.0435 |
| spdim | SPDIM bias | -0.01 | dataset_macro | ALL | NA | 3 | 0.0620 |
| spdim | SPDIM bias | -0.02 | per_dataset | BNCI2014_001 | 1 | 9 | 0.1111 |
| spdim | SPDIM bias | -0.02 | per_dataset | Cho2017 | 1 | 52 | 0.0192 |
| spdim | SPDIM bias | -0.02 | per_dataset | Lee2019_MI | 2 | 54 | 0.0370 |
| spdim | SPDIM bias | -0.02 | subject_weighted | ALL | 4 | 115 | 0.0348 |
| spdim | SPDIM bias | -0.02 | dataset_macro | ALL | NA | 3 | 0.0558 |
