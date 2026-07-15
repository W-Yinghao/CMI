# S2P_41 - FMScope-FSR Bridge Panel 2 SEED-V Interim Result

**Status:** SEED-V SECONDARY PANEL VERIFIED / FACED PRIMARY AND ISRUC DIRECTIONAL PANELS PENDING.

This is a technical result record, not manuscript text. It does not close Panel 2 or authorize downstream
fine-tuning, new pretraining, H4000, CodeBrain, a new dataset, or submission writing.

## Contract closure

All ten immutable representation objects completed the FMScope-aligned pooled `[B,200]` four-arm analysis.
Every object contains 100 same-rank and 100 removed-variance-matched draws for global/source-only and
fresh/exact endpoints. The independent verifier recomputed metrics from persisted `float64` probabilities,
confirmed all 8,000 null rows, checked feature payload hashes and variance matching, and passed 108/108 checks.

The first verifier attempt correctly failed because prediction support had been quantized to `float32` while the
metric CSV retained `float64` values. No tolerance was relaxed; the support payload was regenerated at original
precision and the complete fleet was rerun.

## Secondary inference

SEED-V uses the within-cohort 5/5/5 trial split and cannot license unseen-subject deployment claims.

```text
P2-H1 pooled-high source-only fresh Kappa effect:
  estimate approximately 0
  95% subject-cluster CI [-0.0257, 0.0266]
  same-rank empirical p = 0.941
  variance-matched empirical p = 0.554

P2-H2 global-minus-source fresh Kappa gap:
  estimate = +0.00355
  95% subject-cluster CI [-0.0217, 0.0278]
  same-rank empirical p = 0.287
  variance-matched empirical p = 0.356

P2-H3 high-budget minus H200 exact-head effect:
  estimate = +0.0541 Kappa
  95% subject-cluster CI [0.0177, 0.0922]
  Holm-adjusted p = 0.00420
  disposition = diagnostic only because required task-gate cells do not all pass
```

Only released, H500_s1, H1000_s0, and H1000_s1 pass the frozen pooled-feature task gate. H200, H500_s0, and
both H2000 cells do not. The positive P2-H3 direction therefore cannot support a general exact-head safety or
reliance claim.

## Axis transferability

For the eight trained checkpoints, source-fitted LEACE removes 96.0%-98.6% of test-trial between-subject scatter
(mean 97.7%). Subject decoding after erasure falls close to the 1/16 cohort chance level. Thus the axis remains
highly transferable across SEED-V trial splits while removal produces no identity-specific fresh-head utility.

The licensed interim conclusion is:

> In the SEED-V within-cohort trial setting, a source-fitted subject axis is strongly transferable and removable,
> but higher-budget source-only removal does not outperform generic random removal. A reduction in exact-head
> cost is visible only as a task-gate-limited diagnostic.

FACED remains the primary unseen-subject Panel-2 test. No Panel-2 cross-dataset verdict is issued before FACED and
the scoped ISRUC directional analysis complete.
