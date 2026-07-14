# P13 Completion Report

P13 is complete. The fixed-reservoir Lee2019_MI prevalence stress finished with `162/162` target-seed units, `2,916/2,916` result rows, and `972/972` geometry rows. The final independent red team passed with zero metric, key, subject-aggregation, or bootstrap mismatches.

## Fixed-Prior Geometry EM Prevalence Stress

The experiment used all 54 repaired-split Lee2019_MI targets, source seeds 0/1/2, exact P12 checkpoints, the unchanged balanced evaluation block, and the exact P12 50-trial adaptation reservoir. q=0.1 and q=0.9 were deterministic repeat/crop interventions with class counts `[5,45]` and `[45,5]`; q=0.5 was the exact P12 `[25,25]` center. Methods received ordered EEG/features but no target labels, q value, or target performance.

The primary result does not support the FP-GEM design claim:

`FP-GEM sensitivity minus Joint-GEM sensitivity = -0.0007407 [-0.0036420, +0.0020988]`

The interval crosses zero. Fixed fitting prior did not establish lower prevalence sensitivity than Joint-GEM under the frozen primary test.

All predeclared external comparisons were reported:

- FP-GEM minus RCT: `-0.0061111 [-0.0114198, -0.0010494]`
- FP-GEM minus SPDIM geodesic: `+0.0000617 [-0.0047531, +0.0050000]`
- FP-GEM minus SPDIM bias: `+0.0014815 [-0.0061111, +0.0092593]`

FP-GEM is less sensitive than RCT on the frozen sensitivity endpoint, but not distinguishable from either SPDIM method. This is an external head-to-head result, not the primary claim.

The secondary endpoints show why sensitivity alone is insufficient. FP-GEM endpoint-mean bAcc is `0.6488272 [0.6288272, 0.6688287]`, versus RCT `0.6588889`, SPDIM geodesic `0.6640741`, and SPDIM bias `0.6637654`. FP-GEM is significantly lower than each on endpoint-mean and worst-prevalence bAcc. It moved less than RCT, but did not perform better at the stressed endpoints.

The geometry diagnostic is consistent with the intervention: FP-GEM kept its fitted class-0 prior exactly at the source prior 0.5 for all q. Joint-GEM's mean fitted class-0 prior was 0.4636, 0.4850, and 0.5007 at q=0.1, 0.5, and 0.9. That mechanism did not yield a decisive FP-GEM versus Joint-GEM sensitivity advantage.

## Execution And Provenance

- clean launch commit: `afa21f2e9ae3b448bab271f30a399eb6cad765b0`
- frozen runner SHA-256: `e5b4450e4e8bb9f715fd9ef4e12b6f26d415c9cac4592ab154430f6550903d9e`
- frozen config SHA-256: `12acd01fbad33cdc5feadf2fe54da0c7423960ab6f1bfa7c8a7005ff76b87e2f`
- frozen manifest SHA-256: `8c5b160fcec5ffeaded7faaf196f9753d7e0f7f15e583f8a18a5651ddf1c5802`
- final result SHA-256: `cf9e403eb8be1c0548a95f9007eb7089ee3f93d8bee2401af22587903bffdb2f`
- monitoring: final job absence from `squeue` plus artifact validation; no Slurm accounting command
- stdout: `162/162` complete clean-launch gates
- stderr: 161 empty; one exact post-artifact scheduler handoff verified by job/unit/node/time ordering
- excluded failures: six pre-result density-hash failures, all zero accepted rows with 18 artifacts checksummed
- canceled pending launches: jobs 894872 and 894902, both zero accepted rows

The density-hash failures were hardware-reproduction failures caught before result generation. Exact frozen retries on matching V100 hardware recovered the affected units; no method parameter, checkpoint, split, endpoint, or hyperparameter changed.

## Claim Boundary

P13 is the final experimental decision gate. It does not support the primary claim that FP-GEM is less prevalence-sensitive than Joint-GEM. It supports only the narrower predeclared observation that FP-GEM sensitivity is lower than RCT in this Lee2019_MI fixed-reservoir intervention, while endpoint performance is lower. No equivalence, noninferiority, natural-transfer superiority, broad-benchmark, or post-hoc variant claim is permitted.
