# Step 17 — estimand-consistent harm control

Scope: estimand-consistent harm control (accuracy vs balanced-accuracy); not SOTA. Accuracy-gain and balanced-accuracy-gain are DIFFERENT target functionals; a policy licensed for one does NOT control the other. k>0 is an R2 labeled slice under a sampling contract (class-balanced bAcc requires C13); NOT R1 target-gain identifiability.

- runs: **54** · accuracy-benefit-rate **0.1481** · bAcc-benefit-rate **0.1481** · sign-agreement **1.0**
- runs accuracy-benefit but bAcc-harm: **0** · bAcc-benefit but accuracy-harm: **0**
- accuracy policy controls bAcc: **False** · class-balanced bAcc requires **C13**

| estimand | sampling | k | policy | adapt_cov | harm@adapt | missing_class | missed_benefit |
|---|---|---|---|---:|---:|---:|---:|
| accuracy_gain | iid | 32 | ci_three_way | 0.012 | 0.6821 | 0.0 | 0.9742 |
| accuracy_gain | iid | full | ci_three_way | 0.0 | None | 0.0 | 1.0 |
| accuracy_gain | class_balanced | 32 | ci_three_way | 0.0069 | 0.5699 | 0.0 | 0.98 |
| accuracy_gain | class_balanced | full | ci_three_way | 0.0 | None | 0.0 | 1.0 |
| balanced_accuracy_gain | iid | 32 | ci_three_way | 0.0139 | 0.6765 | 0.0001 | 0.9698 |
| balanced_accuracy_gain | iid | full | ci_three_way | 0.0 | None | 0.0 | 1.0 |
| balanced_accuracy_gain | class_balanced | 32 | ci_three_way | 0.0116 | 0.6538 | 0.0 | 0.973 |
| balanced_accuracy_gain | class_balanced | full | ci_three_way | 0.0 | None | 0.0 | 1.0 |

> Table shows ci_three_way at k in {32, full}, first tau; full grid in the JSON.
> Accuracy-gain and balanced-accuracy-gain are DIFFERENT target functionals; a policy licensed for one does NOT control the other. k>0 is an R2 labeled slice under a sampling contract (class-balanced bAcc requires C13); NOT R1 target-gain identifiability.
