# P12 Completion Report

Status: `COMPLETE` after independent red-team review (`31/31` gates passed).

## Scope

P12 is the frozen two-dataset, three-source-seed same-backbone head-to-head for Fixed-Prior Geometry EM (FP-GEM). It includes BNCI2014_001 and Lee2019_MI, all 63 repaired-split target subjects, source seeds 0/1/2, and exactly the six frozen methods.

The final packet contains 189 target-seed units and 1,134 result rows: 378 new Joint-GEM/FP-GEM rows and 756 same-checkpoint control rows. Every row is `status=ok`; prediction and logits hashes are complete; adaptation and evaluation trial IDs are disjoint; both evaluation classes are present; target labels were not passed to adaptation; no target-performance selection occurred; and the backbone and classifier remained frozen.

## Source Provenance Boundary

P9 persisted source-state hashes but not loadable checkpoint files. Under the pre-result provenance amendment, P12 therefore reproduced the exact P9 source-training configuration and persisted one checkpoint per dataset x target x seed. All six methods in each unit use that same checkpoint.

The P12 source-state hashes do not equal the P9 reference hashes (`0/189` exact state-hash matches), so this is not direct reuse of P9 checkpoints or committed P9 rows. It is a controlled within-P12 same-backbone comparison using exact-P9-configuration source retrains. The four official controls were rerun without tuning so that each control and both GEM methods share the same source state within a unit.

## Mean Results

Evaluation blocks are exactly class-balanced (`[36,36]` for BNCI2014_001 and `[25,25]` for Lee2019_MI), so the reported mean accuracy and balanced accuracy are numerically equal.

| method | BNCI2014_001 acc/bAcc | Lee2019_MI acc/bAcc | subject-weighted acc/bAcc | dataset-macro acc/bAcc |
|---|---:|---:|---:|---:|
| source_only_tsmnet | 0.6096 | 0.5485 | 0.5572 | 0.5790 |
| rct | 0.7274 | 0.6816 | 0.6881 | 0.7045 |
| spdim_geodesic | 0.7274 | 0.6765 | 0.6838 | 0.7020 |
| spdim_bias | 0.7238 | 0.6743 | 0.6814 | 0.6990 |
| Joint-GEM | 0.7094 | 0.6627 | 0.6694 | 0.6860 |
| FP-GEM | 0.7124 | 0.6656 | 0.6723 | 0.6890 |

## Primary Contrasts

Subject-weighted paired balanced-accuracy contrasts after averaging source seeds within subject:

| comparison | estimate | 95% cluster-bootstrap CI |
|---|---:|---:|
| FP-GEM minus source_only_tsmnet | +0.1150 | [+0.0964, +0.1347] |
| FP-GEM minus rct | -0.0159 | [-0.0218, -0.0099] |
| FP-GEM minus spdim_geodesic | -0.0115 | [-0.0170, -0.0059] |
| FP-GEM minus spdim_bias | -0.0091 | [-0.0154, -0.0027] |
| FP-GEM minus Joint-GEM | +0.0029 | [-0.0003, +0.0062] |

Intervals use 10,000 dataset-stratified paired cluster-bootstrap replicates, fixed seed 20260710, and `dataset x target_subject` as the cluster after averaging the three source seeds. The dataset-macro FP-GEM minus Joint-GEM estimate is +0.0030 [+0.0007, +0.0052], but the subject-weighted interval crosses zero and the Lee2019_MI interval also crosses zero. A general fixed-prior superiority claim is therefore not supported.

## Verdict

FP-GEM improves over source-only under the frozen two-dataset grid. It does not outperform RCT, SPDIM geodesic, or SPDIM bias on the subject-weighted primary contrasts. The small FP-GEM versus Joint-GEM difference is not robust across estimands. No equivalence, noninferiority, broad-benchmark, direct-P9-checkpoint-reuse, or target-tuned claim is permitted.

## Execution And Validation

The accepted result fleet used eight final task executions across V100 and A100 resources. Array element `893449` failed before producing an accepted row because the smoke checkpoint reload did not restore custom SPD batch-normalization source-domain buffers; its artifact was archived and excluded. Exact task-0 retry `893456` fresh-trained under the same frozen command and scientific configuration. All excluded launches contain zero accepted rows.

Accepted jobs are absent from `squeue`; all eight final stderr files are empty and all eight stdout files end with `shard_complete`. The independent red-team rebuilt all seed averages and all 10,000-replicate intervals from the final CSV and passed 31/31 completion, provenance, statistical, and claim gates.

The `icml` runtime does not include the `pytest` package. The seven fixture-free tests in `h2cmi/tests/test_fp_gem.py` were therefore invoked directly; all 7/7 passed, covering feature-hook replay, FP/Joint prior behavior, frozen scope, label-access order, source seeding, source-provenance gating, and bootstrap pairing.

Final result CSV SHA-256: `f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d`.
