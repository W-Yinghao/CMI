# OACI EEG-DG Project Memory Through C81P

## Current Gate

```text
C81_AAAI_BASELINE_COMPARISON_PROTOCOL_LOCKED_READY_FOR_PI_AUTHORIZATION
```

C81P is complete. C81E is not authorized. No real C81 baseline result exists.
No training, forward/re-inference, GPU, target-4 primary use, same-label oracle,
BNCI2014_004, seed 5, active acquisition, new feature/kernel/model search, or
manuscript experiment is authorized.

## Accepted Scientific Base

C80E remains the latest scientific result:

```text
C80-A_stable_low_regret_label_budget_frontier_across_training_seeds
B*_seed3 / B*_seed4:       1 / 1 label per class
B=1 regret:                0.353383 / 0.373705
B=1 source-relative gain:  0.426093 / 0.423742
top-1:                     approximately 3.8%
LOTO B* changes:           16 / 16
```

This is a full-panel, source-relative result under the fixed Q0 policy, not
universal one-label sufficiency, low absolute regret, reliable exact-best
localization, target-population stability, deployment, or external validity.

## C81P Operative Objects

```text
protocol commit:          16a0d2eba4715a1cec78da6a79a182fd416a6629
protocol SHA-256:         cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee
method registry SHA-256:  ef48ecf7fcc55188b78b0878d86f07f6239fe4f6c88bbc854829b3a1c7a1a120
implementation commit:    d17ffa62a63b929d36d03f74e4ce79794cd9601b
analysis lock commit:     541651c2ee3343c12d374a7322c91181a860a2c9
analysis lock SHA-256:    b383707f58063c10f719194a995ab34094f6dcefe08c1e71837644db83dc94f1
```

The direct-authorization policy at `3d9dd76` applies: a PI statement naming
C81E is sufficient and does not require a token or repeated hashes. The server
must bind it to the unique current protocol and lock. Authorization remains
milestone-specific.

## Frozen Universe and Views

```text
seeds:                 [3,4]
primary targets:       [1,2,3,5,6,7,8,9]
levels:                [0,1]
candidates/context:    81
primary contexts:      32
primary candidates:    2,592
principal cluster:     target
seed role:             paired training factor
```

Target 4 is engineering-only. Selection must freeze before evaluation labels
open. The same-label oracle is unreachable. Trial ID and row order are
join/split/dependence keys only. Checkpoints, trials, candidate pairs, and Monte
Carlo draws are not independent scientific samples.

## Registered Baselines

The 34-method registry spans fixed controls, strict-source validation,
target-unlabeled confidence/dispersity, normalized logits, optimal transport,
feature geometry, model agreement, source-calibrated target-unlabeled methods,
and frozen construction-label comparators.

The six fixed zero-label primary representatives are ATC, NuclearNorm, MaNo,
COTT, SND, and ALine. Source-validation balanced accuracy is the predeclared
strict-source representative because true Source-LODO retraining folds are not
frozen. The importance-weighting representative remains unavailable because
the necessary frozen density-ratio/discriminator/group-weight objects do not
exist and new fitting is forbidden.

Q0 B=1 is the primary labeled comparator and Q0 FULL is the labeled ceiling.
The complete fixed C80 curve remains mandatory context. Oracle-best is a
denominator only, never a selector.

## Estimands and Inference

Primary endpoints are held-evaluation standardized regret, selected utility,
and source-relative regret gain. Top-1/top-5/top-10 and measurement metrics are
secondary; measurement quality cannot substitute for decision regret.

Q1 asks whether a fixed zero-label representative materially improves over
strict source. Q2 asks whether it is noninferior to Q0 B=1. Both use the fixed
0.05 margin, exact shared-target max-T over 256 sign patterns, at least 6/8
favorable targets, and catastrophic-target guards.

The taxonomy is exhaustive: C81-E blocker, C81-D training-seed/target
heterogeneity, then stable C81-A, C81-B, or C81-C. LOTO stability requires
matching full-panel seed categories and at least 12/16 retained panels.

## Calibration and Integrity

```text
C80 result hashes:             22 / 22 PASS
C80 field/view objects:        11 / 11 PASS
C80 target-level rows:        224 / 224 PASS
C80 LOTO rows:                 16 / 16 PASS
synthetic scenarios:           13 / 13 PASS
familywise checks:              2 / 2 PASS
pair-dependence checks:         2 / 2 PASS
noninferiority checks:          3 / 3 PASS
pre-execution red team:        43 / 43 PASS
final-report red team:         40 / 40 PASS
```

Accepted regression on clean commit `e347a06`: focused 43; C65-C81P 412 plus
one conditional skip and three historical deselections; C23-C81P 823 plus one
and three; full OACI 1,747 plus one and three. All accepted stderr files are
empty. Every superseded or invalid attempt is retained in the regression
ledger.

## Next Control Point

Stop for PI review. C81E may run only after direct authorization. It is a
read-only existing-field comparative audit and does not authorize any later
milestone or expanded scientific scope.
