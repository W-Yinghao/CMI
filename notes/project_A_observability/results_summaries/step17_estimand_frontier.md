# Step 17 — per-estimand harm-control frontier

Scope: per-estimand harm-control frontier (accuracy vs balanced-accuracy); not SOTA. Accuracy-gain and balanced-accuracy-gain frontiers are kept SEPARATE; there is no overall best policy across estimands (different target functionals). The balanced_accuracy_gain:class_balanced frontier requires contract C13. Frontiers are R2 label-budget views, not R1 target-gain identifiability.

- no overall best across estimands: **True** · accuracy policy controls bAcc: **False** · class-balanced bAcc requires **C13**

| group | requires_contract | meets 0.05 | meets 0.10 | meets 0.20 |
|---|---|---|---|---|
| accuracy_gain:iid | None | True | True | True |
| accuracy_gain:class_balanced | None | True | True | True |
| balanced_accuracy_gain:iid | None | True | True | True |
| balanced_accuracy_gain:class_balanced | C13 | True | True | True |

> Accuracy-gain and balanced-accuracy-gain frontiers are kept SEPARATE; there is no overall best policy across estimands (different target functionals). The balanced_accuracy_gain:class_balanced frontier requires contract C13. Frontiers are R2 label-budget views, not R1 target-gain identifiability.
