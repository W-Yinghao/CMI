# FP-GEM Same-Backbone Head-to-Head

Status: frozen P12 two-dataset, three-source-seed comparison. All six methods in each unit share one exact-P9-configuration source retrain; this is not a broad benchmark.

## Mean Balanced Accuracy

| method | BNCI2014_001 | Lee2019_MI | subject-weighted | dataset-macro |
|---|---:|---:|---:|---:|
| source_only_tsmnet | 0.6096 [0.5468, 0.6790] | 0.5485 [0.5360, 0.5621] | 0.5572 [0.5433, 0.5724] | 0.5790 [0.5471, 0.6145] |
| rct | 0.7274 [0.6445, 0.8015] | 0.6816 [0.6552, 0.7091] | 0.6881 [0.6624, 0.7141] | 0.7045 [0.6612, 0.7442] |
| spdim_geodesic | 0.7274 [0.6430, 0.8014] | 0.6765 [0.6509, 0.7035] | 0.6838 [0.6584, 0.7093] | 0.7020 [0.6582, 0.7417] |
| spdim_bias | 0.7238 [0.6384, 0.7984] | 0.6743 [0.6485, 0.7016] | 0.6814 [0.6564, 0.7068] | 0.6990 [0.6546, 0.7386] |
| Joint-GEM | 0.7094 [0.6384, 0.7788] | 0.6627 [0.6385, 0.6881] | 0.6694 [0.6461, 0.6933] | 0.6860 [0.6481, 0.7229] |
| FP-GEM | 0.7124 [0.6404, 0.7824] | 0.6656 [0.6410, 0.6914] | 0.6723 [0.6488, 0.6966] | 0.6890 [0.6504, 0.7265] |

## Primary Paired Balanced-Accuracy Contrasts

| comparison | BNCI2014_001 | Lee2019_MI | subject-weighted | dataset-macro |
|---|---:|---:|---:|---:|
| FP-GEM minus source_only_tsmnet | +0.1029 [+0.0453, +0.1636] | +0.1170 [+0.0973, +0.1377] | +0.1150 [+0.0964, +0.1347] | +0.1100 [+0.0799, +0.1427] |
| FP-GEM minus rct | -0.0149 [-0.0298, +0.0010] | -0.0160 [-0.0225, -0.0095] | -0.0159 [-0.0218, -0.0099] | -0.0155 [-0.0236, -0.0067] |
| FP-GEM minus spdim_geodesic | -0.0149 [-0.0319, +0.0031] | -0.0110 [-0.0167, -0.0051] | -0.0115 [-0.0170, -0.0059] | -0.0130 [-0.0221, -0.0033] |
| FP-GEM minus spdim_bias | -0.0113 [-0.0314, +0.0103] | -0.0088 [-0.0151, -0.0020] | -0.0091 [-0.0154, -0.0027] | -0.0100 [-0.0206, +0.0013] |
| FP-GEM minus Joint-GEM | +0.0031 [+0.0005, +0.0051] | +0.0028 [-0.0009, +0.0067] | +0.0029 [-0.0003, +0.0062] | +0.0030 [+0.0007, +0.0052] |

Accuracy summaries and contrasts are included in the machine-readable packet. Source seeds are averaged within subject/method before every aggregate. Intervals are 10,000-replicate dataset-stratified paired cluster-bootstrap intervals with seed 20260710.

P9 did not persist source checkpoint weights. Accordingly, committed P9 rows are provenance references rather than direct inputs here; the four frozen official controls were rerun without tuning from the same persisted unit checkpoint used by Joint-GEM and FP-GEM.

No equivalence, noninferiority, broad-benchmark, target-selection, or third-dataset claim is permitted. Interpret estimates only under the precommitted grid in `FP_GEM_METHOD_FREEZE.md`.

Final result CSV SHA-256: `f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d`.
