# Step 12 — retrospective offline-TTA harm predictor

Scope: retrospective harm prediction; not SOTA. empirical retrospective predictor over audited cells; NOT target-gain/harm identifiability (TOS-1/TU-2 hold). Oracle harm is the target, never a feature.

- runs: **54** · harm-rate **0.8519** · majority-baseline balanced-acc **0.5** · minority-class n **8** (LOTO is noisy at this power)
- CV: leave-one-(dataset,target_subject)-out logistic regression (class_weight=balanced)

- **R0_source_only**: 4 features · harm-pred balanced-acc **0.3342** · AUC **0.1793** · beats-baseline **False** · perm-null p95 **0.6608** · beats-perm-null **False**
- **R1_target_unlabeled**: 19 features · harm-pred balanced-acc **0.6522** · AUC **0.7255** · beats-baseline **True** · perm-null p95 **0.6413** · beats-perm-null **True**
- **R1 − R0 balanced-acc delta: 0.318**
- **verdict: marginal_at_permutation_boundary_not_robust** · any predictor beats baseline: **True**
- oracle never a feature: **True**

> F2(k) = F1 + a k-label target slice is studied on the controlled simulator in minimal_paired.py, not here (real runs store no per-trial target labels).

> Balanced-acc near 0.5 = no retrospective signal beyond the majority class; any lift is an empirical retrospective predictor, not identifiability.
