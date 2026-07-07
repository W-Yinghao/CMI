# Step 12 — retrospective offline-TTA harm predictor

Scope: retrospective harm prediction; not SOTA. empirical retrospective predictor over audited cells; NOT target-gain/harm identifiability (TOS-1/TU-2 hold). Oracle harm is the target, never a feature.

- runs: **54** · harm-rate **0.8333** · majority-baseline balanced-acc **0.5** · minority-class n **9** (LOTO is noisy at this power)
- CV: leave-one-(dataset,target_subject)-out logistic regression (class_weight=balanced)

- **R0_source_only**: 4 features · harm-pred balanced-acc **0.2556** · AUC **0.1778** · beats-baseline **False**
- **R1_target_unlabeled**: 11 features · harm-pred balanced-acc **0.4222** · AUC **0.2173** · beats-baseline **False**
- **R1 − R0 balanced-acc delta: 0.1666**
- **verdict: no_retrospective_harm_signal_above_baseline** · any predictor beats baseline: **False**
- oracle never a feature: **True**

> F2(k) = F1 + a k-label target slice is studied on the controlled simulator in minimal_paired.py, not here (real runs store no per-trial target labels).

> Balanced-acc near 0.5 = no retrospective signal beyond the majority class; any lift is an empirical retrospective predictor, not identifiability.
