# Official SPDIM W1 Repaired-Split Three-Seed Result Digest

Label: Official SPDIM W1 repaired-split three-source-seed same-split baseline.

- rows: `1380/1380`
- final_result_sha256: `95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3`
- bootstrap: `10000` dataset-stratified subject-cluster replicates, seed `20260710`
- seeds are averaged within dataset x target subject x method before aggregation.

## Method bAcc

| scope | dataset | source_only_tsmnet | rct | spdim_geodesic | spdim_bias |
|---|---|---:|---:|---:|---:|
| per_dataset | BNCI2014_001 | 0.6096 [0.5478, 0.6790] | 0.7269 [0.6425, 0.8025] | 0.7279 [0.6435, 0.8040] | 0.7233 [0.6373, 0.7989] |
| per_dataset | Cho2017 | 0.5235 [0.5158, 0.5325] | 0.5976 [0.5782, 0.6195] | 0.5960 [0.5766, 0.6178] | 0.5969 [0.5773, 0.6188] |
| per_dataset | Lee2019_MI | 0.5485 [0.5362, 0.5617] | 0.6816 [0.6551, 0.7084] | 0.6772 [0.6510, 0.7035] | 0.6743 [0.6484, 0.7007] |
| subject_weighted | ALL | 0.5420 [0.5334, 0.5509] | 0.6472 [0.6304, 0.6638] | 0.6444 [0.6277, 0.6610] | 0.6432 [0.6265, 0.6599] |
| dataset_macro | ALL | 0.5605 [0.5394, 0.5841] | 0.6687 [0.6393, 0.6965] | 0.6670 [0.6374, 0.6950] | 0.6648 [0.6345, 0.6929] |

## Paired bAcc Contrasts

| contrast | subject-weighted estimate [95% CI] | dataset-macro estimate [95% CI] |
|---|---:|---:|
| rct_minus_source_only_tsmnet | +0.1052 [+0.0918, +0.1190] | +0.1082 [+0.0853, +0.1326] |
| spdim_geodesic_minus_source_only_tsmnet | +0.1024 [+0.0895, +0.1161] | +0.1065 [+0.0840, +0.1305] |
| spdim_bias_minus_source_only_tsmnet | +0.1012 [+0.0883, +0.1149] | +0.1043 [+0.0816, +0.1287] |
| spdim_geodesic_minus_rct | -0.0027 [-0.0047, -0.0008] | -0.0017 [-0.0036, +0.0002] |
| spdim_bias_minus_rct | -0.0040 [-0.0073, -0.0007] | -0.0039 [-0.0076, -0.0001] |

## Subject-Weighted Harm Rate at Delta < 0

| contrast | count / 115 | rate |
|---|---:|---:|
| rct_minus_source_only_tsmnet | 6 / 115 | 0.0522 |
| spdim_geodesic_minus_source_only_tsmnet | 6 / 115 | 0.0522 |
| spdim_bias_minus_source_only_tsmnet | 6 / 115 | 0.0522 |
| spdim_geodesic_minus_rct | 61 / 115 | 0.5304 |
| spdim_bias_minus_rct | 60 / 115 | 0.5217 |

## Seed Stability (bAcc)

| dataset | method | seed 0 | seed 1 | seed 2 | range | population SD | ranks 0/1/2 |
|---|---|---:|---:|---:|---:|---:|---|
| BNCI2014_001 | source_only_tsmnet | 0.6265 | 0.5772 | 0.6250 | 0.0494 | 0.0229 | 4/4/4 |
| BNCI2014_001 | rct | 0.7299 | 0.7253 | 0.7253 | 0.0046 | 0.0022 | 1/3/1 |
| BNCI2014_001 | spdim_geodesic | 0.7253 | 0.7361 | 0.7222 | 0.0139 | 0.0060 | 2/1/2 |
| BNCI2014_001 | spdim_bias | 0.7222 | 0.7315 | 0.7160 | 0.0154 | 0.0063 | 3/2/3 |
| Cho2017 | source_only_tsmnet | 0.5286 | 0.5165 | 0.5254 | 0.0120 | 0.0051 | 4/4/4 |
| Cho2017 | rct | 0.5998 | 0.6012 | 0.5918 | 0.0094 | 0.0041 | 1/1/2 |
| Cho2017 | spdim_geodesic | 0.5992 | 0.5997 | 0.5890 | 0.0107 | 0.0049 | 2/2/3 |
| Cho2017 | spdim_bias | 0.5984 | 0.5972 | 0.5951 | 0.0033 | 0.0014 | 3/3/1 |
| Lee2019_MI | source_only_tsmnet | 0.5500 | 0.5444 | 0.5511 | 0.0067 | 0.0029 | 4/4/4 |
| Lee2019_MI | rct | 0.6822 | 0.6815 | 0.6811 | 0.0011 | 0.0005 | 1/1/1 |
| Lee2019_MI | spdim_geodesic | 0.6793 | 0.6730 | 0.6793 | 0.0063 | 0.0030 | 2/2/3 |
| Lee2019_MI | spdim_bias | 0.6715 | 0.6719 | 0.6796 | 0.0081 | 0.0038 | 3/3/2 |

## Interpretation

SPDIM-specific claims are determined by the paired `spdim_geodesic_minus_rct` and `spdim_bias_minus_rct` rows above. No equivalence or noninferiority claim is made, and no post-hoc margin is introduced.

## Internal Validation Review

- Seed rows are not treated as independent biological units.
- P8 seed-0 bytes remain unchanged and seed-0 rows are copied, not recomputed.
- The P100 launch contributes zero rows; all accepted shards pass real-failure log screening.
