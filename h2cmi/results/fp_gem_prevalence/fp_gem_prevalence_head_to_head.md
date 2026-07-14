# FP-GEM Fixed-Reservoir Prevalence Stress

Status: P13 final targeted Lee2019_MI prevalence intervention. This is not a natural-transfer benchmark or a hyperparameter search.

## Prevalence Sensitivity

Lower is better. Source seeds are averaged within each target subject before the paired subject bootstrap.

| method | sensitivity |
|---|---:|
| source_only_tsmnet | 0.0000 [0.0000, 0.0000] |
| rct | 0.0357 [0.0279, 0.0447] |
| spdim_geodesic | 0.0295 [0.0244, 0.0351] |
| spdim_bias | 0.0281 [0.0228, 0.0341] |
| Joint-GEM | 0.0303 [0.0238, 0.0375] |
| FP-GEM | 0.0296 [0.0233, 0.0367] |

## Frozen Comparisons

| comparison | paired difference | 95% CI entirely below zero |
|---|---:|---:|
| FP-GEM minus Joint-GEM | -0.0007 [-0.0036, +0.0021] | false |
| FP-GEM minus rct | -0.0061 [-0.0114, -0.0010] | true |
| FP-GEM minus spdim_geodesic | +0.0001 [-0.0048, +0.0050] | false |
| FP-GEM minus spdim_bias | +0.0015 [-0.0061, +0.0093] | false |

The primary FP-GEM design claim uses only `FP-GEM minus Joint-GEM` and is supported only when the full percentile 95% CI is below zero. The three external comparisons are reported separately and are not selected post hoc.

## Secondary Endpoints

| method | endpoint mean bAcc | worst-prevalence bAcc | endpoint disagreement |
|---|---:|---:|---:|
| source_only_tsmnet | 0.5485 [0.5363, 0.5620] | 0.5485 [0.5363, 0.5620] | 0.0000 [0.0000, 0.0000] |
| rct | 0.6589 [0.6388, 0.6787] | 0.6406 [0.6216, 0.6596] | 0.1328 [0.1174, 0.1488] |
| spdim_geodesic | 0.6641 [0.6427, 0.6862] | 0.6451 [0.6242, 0.6669] | 0.1254 [0.1123, 0.1387] |
| spdim_bias | 0.6638 [0.6410, 0.6877] | 0.6462 [0.6240, 0.6702] | 0.1120 [0.0984, 0.1264] |
| Joint-GEM | 0.6470 [0.6275, 0.6666] | 0.6274 [0.6084, 0.6464] | 0.1233 [0.1086, 0.1388] |
| FP-GEM | 0.6488 [0.6288, 0.6688] | 0.6288 [0.6095, 0.6481] | 0.1248 [0.1098, 0.1404] |

## Mean bAcc By Prevalence

| method | q=0.1 | q=0.5 | q=0.9 |
|---|---:|---:|---:|
| source_only_tsmnet | 0.5485 [0.5363, 0.5620] | 0.5485 [0.5363, 0.5620] | 0.5485 [0.5363, 0.5620] |
| rct | 0.6646 [0.6423, 0.6868] | 0.6816 [0.6556, 0.7085] | 0.6532 [0.6337, 0.6727] |
| spdim_geodesic | 0.6691 [0.6452, 0.6938] | 0.6765 [0.6511, 0.7030] | 0.6590 [0.6384, 0.6805] |
| spdim_bias | 0.6683 [0.6436, 0.6940] | 0.6743 [0.6490, 0.7010] | 0.6593 [0.6365, 0.6837] |
| Joint-GEM | 0.6557 [0.6348, 0.6775] | 0.6627 [0.6389, 0.6877] | 0.6383 [0.6180, 0.6585] |
| FP-GEM | 0.6583 [0.6362, 0.6812] | 0.6656 [0.6414, 0.6906] | 0.6394 [0.6194, 0.6594] |

## Geometry Diagnostics

| method | q | mean geometry displacement from q=0.5 | mean fitted class-0 prior |
|---|---:|---:|---:|
| Joint-GEM | 0.1 | 1.4357 | 0.4636 |
| Joint-GEM | 0.5 | 0.0000 | 0.4850 |
| Joint-GEM | 0.9 | 1.4244 | 0.5007 |
| FP-GEM | 0.1 | 1.4144 | 0.5000 |
| FP-GEM | 0.5 | 0.0000 | 0.5000 |
| FP-GEM | 0.9 | 1.4040 | 0.5000 |

The evaluation block is unchanged and balanced. Target labels were used only by the offline intervention builder; adaptation methods received ordered EEG/features without labels or q. q=0.5 is an exact hash-gated P12 replay/reuse center.

No equivalence, noninferiority, broad-benchmark, or natural-transfer superiority claim is made.

Final result CSV SHA-256: `cf9e403eb8be1c0548a95f9007eb7089ee3f93d8bee2401af22587903bffdb2f`.
