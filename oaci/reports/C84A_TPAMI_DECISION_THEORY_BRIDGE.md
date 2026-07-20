# C84A TPAMI Decision-Theory Bridge

## Status And Boundary

This is a post-C84S read-only synthesis. Every empirical summary is
`POST_C84S_EXPLORATORY_DESCRIPTIVE`. It neither changes C84-D/C84-L4 nor proves
an information-ordering, minimax, or transport theorem.

## Information, Policy, And Risk

Let `E` denote an information experiment, `D(E)` the unrestricted set of
decision rules measurable under that experiment, `Delta(E)` the frozen
registered policy class, and `R(delta; E)` the risk under the common action and
loss definition. Distinguish:

```text
unrestricted optimal risk:
  R*(E) = inf_{delta in D(E)} R(delta; E)

registered-policy optimal risk:
  R_Delta(E) = inf_{delta in Delta(E)} R(delta; E)

policy approximation/optimization gap:
  G_Delta(E) = R_Delta(E) - R*(E) >= 0
```

If one experiment Blackwell-dominates another and the state, action and loss
spaces are common, unrestricted optimal risk cannot increase. C84S does not
estimate those unrestricted infima. It evaluates particular pre-registered,
non-nested policies: fixed zero-label formulas and the fixed Q0 construction-
label policy. Therefore an observed row in which COTT has lower regret than Q0
B=1 or Q0 FULL cannot establish that unlabeled observations are more
informative than labels.

The observed ordering can arise because the registered policy classes differ,
because a policy uses its information imperfectly, because optimization is
restricted, or because the robust target-level gate differs from mean risk.
Q0 failure can therefore be evidence about registered policy actionability and
tail robustness without being evidence that the labels carry no information.

## Action Geometry

The compact result separates four objects:

1. Global ranking fidelity: Spearman, Kendall, and pairwise ordering.
2. Top-tail localization: top-1, top-5, and top-10.
3. Selected-action regret: loss from the frozen selected candidate.
4. Robust qualification: mean, multiplicity, target-count, tail, panel/seed,
   level, and LOTO gates.

Cho MaNo illustrates this separation: near-zero global rank association and
low top-1 coexist with a Q1/Q2-qualified selected action. The allowed compact
tables show that its selections concentrate in the ERM regime, but they do not
contain the full candidate utility geometry needed to identify near-optimal
action density or a causal mechanism.

## What Remains Unidentified

- Blackwell or Le Cam deficiency between the information experiments.
- The unrestricted value of construction labels.
- The policy approximation gap of Q0, COTT, or MaNo.
- A minimax or CVaR-optimal selector.
- Causal effects of source-support deletion on candidate geometry.
- Transport beyond the three harmonized binary-MI cohorts and frozen zoo.

The evidence-to-theory mapping is frozen in
`c84a_tables/information_policy_action_geometry_matrix.csv`; prospective gaps
and empirical requirements are in `theory_gap_registry.csv` and
`next_experiment_decision_matrix.csv`. No next experiment is authorized.
