# C86P Protocol Timing Audit

## Known Before Lock

The C84/C85 scientific outcomes, C85E exploratory summaries, and C85 theorem
statuses were known before C86P. Public documentation, the installed MOABB
imagery catalog, and loader source were inspected only as metadata. The audit
used no published benchmark performance to include or exclude a cohort.

The metadata audit identified three canonical untouched cohorts satisfying the
same prospective interface rule: `Brandl2020`, `Dreyer2023`, and `Kumar2024`.
All three are included. `BNCI2014_004` remains a separately classified stress
candidate and fails the common minimum-subject rule; it was not hand-selected.

Before this protocol commit, C86P did not download or open new EEG, read a new
target label or candidate output, train or forward a candidate, run an active
policy, create a C84 trial-level loss-vector field, use a GPU, or inspect a C86
scientific result.

## Prospectively Fixed

This commit fixes before any C86 protected or confirmation access:

```text
the untouched-cohort eligibility and all-cohorts rule;
the 81-action ERM/OACI/SRC structure and canonical tie rule;
the label-blind acquisition/evaluation split;
total-query budgets [4,8,16,32,FULL];
passive uniform as the primary equal-budget comparator;
the active-method families, common probability floor, and 2,048 chains;
the sequential without-replacement LURE weights;
the historical composite-utility plug-in selection rule;
the distinction between unbiased linear moments and nonlinear plug-in outputs;
target-subject aggregation, max-T family, and qualification thresholds;
mean, worst-target, CVaR, and epsilon-near-optimal endpoints;
C86-A--E and C86-L1--L4 semantics;
development/confirmation separation and all claim boundaries.
```

`FULL` is fixed as a common all-acquisition-label reference, not as an active
versus passive superiority test. C86P creates no real-data execution lock. Each
future C86 stage requires an additive scope-specific lock and fresh direct PI
authorization.
