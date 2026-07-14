# Final FP-GEM Head-to-Head

**Same-backbone, same-checkpoint comparison on two repaired-split motor-imagery datasets. FP-GEM improves over source-only but does not outperform the RCT or SPDIM baselines.**

| method | BNCI2014-001 bAcc | Lee2019-MI bAcc | subject-weighted bAcc | dataset-macro bAcc | delta vs source-only | FP-GEM minus method, paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| source_only_tsmnet | 0.6096 | 0.5485 | 0.5572 | 0.5790 | +0.0000 | +0.1150 [+0.0964, +0.1347] |
| RCT | 0.7274 | 0.6816 | 0.6881 | 0.7045 | +0.1309 | -0.0159 [-0.0218, -0.0099] |
| SPDIM geodesic | 0.7274 | 0.6765 | 0.6838 | 0.7020 | +0.1266 | -0.0115 [-0.0170, -0.0059] |
| SPDIM bias | 0.7238 | 0.6743 | 0.6814 | 0.6990 | +0.1241 | -0.0091 [-0.0154, -0.0027] |
| Joint-GEM | 0.7094 | 0.6627 | 0.6694 | 0.6860 | +0.1121 | +0.0029 [-0.0003, +0.0062] |
| FP-GEM | 0.7124 | 0.6656 | 0.6723 | 0.6890 | +0.1150 | reference |

The delta column is the subject-weighted bAcc difference from `source_only_tsmnet`. The five paired intervals compare FP-GEM with the method in that row after averaging the three source seeds within each of 63 target subjects. Intervals use the frozen P12 10,000-replicate dataset-stratified cluster bootstrap with seed 20260710.

This is the P12 same-backbone comparison, not a broad benchmark. Source: `h2cmi/results/fp_gem_main/fp_gem_results.csv`, SHA-256 `f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d`.
