# Step 18 — TTA Harm Mechanisms and Target-Prior Stress

## Motivation

Step 17 showed that accuracy gain and balanced-accuracy gain coincide on the current grid because every
target set is class-balanced.

This does not mean prior is irrelevant. It means the current evaluation prior is uniform.

A deployment prior or operating utility can differ from the benchmark prior. Then the same class-wise
recall deltas can produce different gain signs.

## Questions

- **Q1.** Is offline-TTA harm global, or class-specific?
- **Q2.** Which true classes lose recall after adaptation?
- **Q3.** Which identity→adapt prediction transitions cause harm?
- **Q4.** Are some runs prior-dependent, meaning the sign of adaptation gain changes under a different
  class prior?
- **Q5.** How much of the current negative result is robust to all priors versus tied to the benchmark
  uniform prior?

## Method (exact, non-neural)

A prior-weighted gain is a linear functional of the per-class recall deltas:

```
gain(π) = Σ_c π_c · recall_delta_c        (balanced accuracy = uniform-π case)
```

Over the simplex the extreme gains are the extreme class deltas, so the sign is prior-dependent iff
`min_c recall_delta_c < 0 < max_c recall_delta_c`. This is read straight off the Step-18 harm-mechanism
decomposition — no retraining, no new model.

## Regime boundary

All class-wise gain and prediction-transition analyses use target labels. They are oracle/evaluation-only
mechanism analyses.

Prior stress tests are counterfactual evaluations of the same predictions under declared operating priors
(contract **C14**). They do NOT identify the actual target prior under R0 or R1 (that needs TU-1). C14 is
a *declared* operating condition, never a source-only target-prior estimate — that remains the
Prior-Decoupled boundary ([04_prior_decoupled_theory.md](../04_prior_decoupled_theory.md)).

## Results

See [14_harm_mechanism_results.md](14_harm_mechanism_results.md) (harm channels) and
[15_prior_stress_results.md](15_prior_stress_results.md) (prior stress). Headline: on this grid TTA harm
is almost entirely **class-specific and prior-dependent**, not global — only ~4% of runs are harmful
under *all* priors, ~96% are prior-dependent — yet this does not make adaptation safe (the class deltas
are oracle and the true deployment prior is unidentified under R0/R1).

## Non-goals

- no new datasets
- no retraining
- no SOTA claim
- no manuscript writing
