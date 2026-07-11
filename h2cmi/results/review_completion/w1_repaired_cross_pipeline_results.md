# Standardized Repaired-W1 Cross-Pipeline Results

All intervals in this file are labeled `posthoc_cross_pipeline_comparability_audit`. They are post-hoc comparability intervals, not preregistered primary intervals.

Seeds 0/1/2 are averaged within dataset x target_subject x method before aggregation. Bootstrap: 10,000 replicates, seed 20260710, dataset-stratified target-subject clusters.

## H2CMI repaired W1

| method | BNCI2014-001 | Cho2017 | Lee2019-MI | subject-weighted | delta vs own baseline | dataset-macro |
|---|---:|---:|---:|---:|---:|---:|
| identity, uniform decision prior | 0.6893 [0.6044, 0.7762] | 0.6338 [0.6119, 0.6574] | 0.7094 [0.6819, 0.7370] | 0.6736 [0.6558, 0.6914] | +0.0000 [+0.0000, +0.0000] | 0.6775 [0.6474, 0.7090] |
| pooled | 0.7197 [0.6435, 0.7963] | 0.6402 [0.6176, 0.6642] | 0.7222 [0.6926, 0.7520] | 0.6849 [0.6664, 0.7033] | +0.0113 [+0.0058, +0.0168] | 0.6940 [0.6658, 0.7226] |
| FRSC | 0.7078 [0.6291, 0.7891] | 0.6417 [0.6195, 0.6654] | 0.7183 [0.6890, 0.7479] | 0.6828 [0.6645, 0.7012] | +0.0092 [+0.0046, +0.0138] | 0.6893 [0.6601, 0.7195] |
| fixed-prior iterative, uniform decision prior | 0.7135 [0.6307, 0.7953] | 0.6402 [0.6176, 0.6641] | 0.7222 [0.6932, 0.7515] | 0.6844 [0.6659, 0.7029] | +0.0108 [+0.0055, +0.0162] | 0.6920 [0.6622, 0.7223] |
| joint-fit geometry, uniform decision prior | 0.7022 [0.6193, 0.7876] | 0.6405 [0.6178, 0.6647] | 0.7174 [0.6877, 0.7475] | 0.6814 [0.6628, 0.7003] | +0.0078 [+0.0026, +0.0133] | 0.6867 [0.6563, 0.7182] |
| Latent-IM-Diag | 0.7119 [0.6286, 0.7953] | 0.6401 [0.6173, 0.6641] | 0.7199 [0.6906, 0.7495] | 0.6832 [0.6645, 0.7017] | +0.0096 [+0.0042, +0.0150] | 0.6906 [0.6605, 0.7214] |

## Official SPDIM repaired W1

| method | BNCI2014-001 | Cho2017 | Lee2019-MI | subject-weighted | delta vs own baseline | dataset-macro |
|---|---:|---:|---:|---:|---:|---:|
| source-only TSMNet | 0.6096 [0.5478, 0.6790] | 0.5235 [0.5158, 0.5325] | 0.5485 [0.5362, 0.5617] | 0.5420 [0.5334, 0.5509] | +0.0000 [+0.0000, +0.0000] | 0.5605 [0.5394, 0.5841] |
| RCT | 0.7269 [0.6425, 0.8025] | 0.5976 [0.5782, 0.6195] | 0.6816 [0.6551, 0.7084] | 0.6472 [0.6304, 0.6638] | +0.1052 [+0.0918, +0.1190] | 0.6687 [0.6393, 0.6965] |
| SPDIM geodesic | 0.7279 [0.6435, 0.8040] | 0.5960 [0.5766, 0.6178] | 0.6772 [0.6510, 0.7035] | 0.6444 [0.6277, 0.6610] | +0.1024 [+0.0895, +0.1161] | 0.6670 [0.6374, 0.6950] |
| SPDIM bias | 0.7233 [0.6373, 0.7989] | 0.5969 [0.5773, 0.6188] | 0.6743 [0.6484, 0.7007] | 0.6432 [0.6265, 0.6599] | +0.1012 [+0.0883, +0.1149] | 0.6648 [0.6345, 0.6929] |

## Claim Boundary

Absolute values may be compared only as a same-split full-pipeline diagnostic. Each delta is paired only against that pipeline's own baseline. The table does not isolate an adapter effect across pipelines.
