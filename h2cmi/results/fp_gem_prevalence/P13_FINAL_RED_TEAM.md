# P13 Final Red-Team Review

Status: `PASS_WITH_PRIMARY_CLAIM_NOT_SUPPORTED`
Machine-readable review: `P13_FINAL_RED_TEAM.json`

## Independent Checks

- Reconstructed all 162 repaired-split Lee2019_MI evaluation blocks from the frozen trial IDs; every block has class counts `[25,25]`.
- Recomputed acc, balanced accuracy, and macro-F1 from evaluation labels and all 2,916 persisted prediction vectors: `0` mismatches.
- Bound all 2,916 final CSV rows and 972 geometry rows back to raw target/seed/q/method artifacts: complete keys, no duplicates, `0` binding mismatches.
- Recomputed all 324 seed-averaged subject-method rows, the frozen sensitivity and secondary endpoints, all fixed paired contrasts, and the 10,000-replicate subject bootstrap with seed 20260710: `0` mismatches.
- Confirmed 162/162 exact q=0.5 P12 centers, complete prediction/logits hashes, exact P12 checkpoints, no split overlap, no labels or q passed to methods, no target-performance selection, and no fresh source training.
- Confirmed all recorded jobs absent from `squeue`; 162 accepted stdout gates pass; stderr is 161 empty plus one precisely verified post-artifact scheduler handoff.
- Verified six failed pre-result attempts and 18 excluded artifacts by checksum; both pending canceled launches contributed zero accepted rows.

## Scientific Verdict

Primary comparison, lower sensitivity is better:

`FP-GEM minus Joint-GEM = -0.0007407 [-0.0036420, +0.0020988]`

The CI crosses zero, so the frozen primary FP-GEM design claim is **not supported**.

Predeclared external sensitivity comparisons:

- FP-GEM minus RCT: `-0.0061111 [-0.0114198, -0.0010494]`
- FP-GEM minus SPDIM geodesic: `+0.0000617 [-0.0047531, +0.0050000]`
- FP-GEM minus SPDIM bias: `+0.0014815 [-0.0061111, +0.0092593]`

FP-GEM is less sensitive than RCT on this endpoint, but is not distinguishable from either SPDIM variant. This external result does not rescue the primary claim and is not a selected comparison.

The secondary endpoints prevent a broader robustness claim. FP-GEM endpoint-mean bAcc is lower than RCT by `-0.0100617 [-0.0153086, -0.0045062]`, lower than SPDIM geodesic by `-0.0152469 [-0.0217284, -0.0088873]`, and lower than SPDIM bias by `-0.0149383 [-0.0246312, -0.0055556]`. Its worst-prevalence bAcc is also lower than all three. Reduced sensitivity versus RCT therefore means less movement around q=0.5, not better endpoint performance.

FP-GEM versus Joint-GEM endpoint-mean bAcc is `+0.0018519 [-0.0006173, +0.0043827]`; this is uncertain. The fitted FP-GEM class-0 prior remains exactly 0.5 across q, while the descriptive mean Joint-GEM fitted prior moves from 0.4636 at q=0.1 to 0.5007 at q=0.9. No equivalence, noninferiority, broad-benchmark, or natural-transfer superiority claim is supported.

## Residual Limit

New-q logits vectors were not persisted, so their hashes cannot be independently regenerated. Their hashes are complete, and all q=0.5 logits reproduce P12 exactly. Prediction vectors, which determine every reported classification metric, were persisted and independently recomputed without error.

Final result SHA-256: `cf9e403eb8be1c0548a95f9007eb7089ee3f93d8bee2401af22587903bffdb2f`.
