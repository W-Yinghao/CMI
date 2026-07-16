# C85R Synthetic Contract Semantic Repair Readiness

## Final Gate

```text
C85_SYNTHETIC_CONTRACT_V2_SEMANTICALLY_REPAIRED_READY_FOR_C85T_PM_REVIEW
```

C85R is an additive no-real-data contract repair. It publishes no theorem,
counterexample result, 4,096-replicate synthetic result, empirical gate,
active-acquisition lock, real-data authorization, or manuscript text.

## Chronology And Identity

```text
C85P final HEAD entering repair:
  00146d25fc15c56c771c87062bb1f2d2f262f59a

C85R protocol-before-V2 commit:
  03bb684e59e3432ae6f484c8c8a537213f52a6cd

C85R implementation commit:
  360c422f3110d61cfef5d09fead78562bb52c497

C85R repair-protocol SHA-256:
  e37bb444fdd174ba4ca1f95e91d9193378f11dd0ef2aeac3e03cbf6249a34b68

historical V1 generator SHA-256:
  c87fec6a6572291fad8849f6c08bea2cb3f49467e243ded1d44c1f38e3d0b297

operative V2 generator SHA-256:
  e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a
```

The repair protocol and timing audit were committed and pushed before the V2
generator, validator, tables, or tests existed. The historical C85P protocol,
V1 generator, 32 registries, readiness report, red team, and regression report
remain byte-identical.

## Historical Blocker

The V1 generator was schema-valid but not ready for C85T execution:

```text
S10:
  required a strict registered-policy reversal that its fixed policies did not
  produce

S9:
  did not define a joint full-information loss-vector law

S6/S7:
  did not define the stochastic utility-estimation error law or its coupling
```

The prior C85P success gate is therefore superseded by this semantic audit. The
accepted state/experiment/action/risk foundation is unchanged.

## S10 Exact Repair

Exact finite enumeration gives:

| Object | Policy | Exact risk |
|---|---|---:|
| coarse unrestricted/registered | action 1 at `y0` and `y1` | `11/40` |
| historical rich registered | always action 1 | `11/40` |
| V2 rich unrestricted | statewise optimum | `0` |
| V2 rich registered | always action 0 | `3/5` |
| V2 rich policy gap | registered minus unrestricted | `3/5` |
| V2 registered reversal | rich minus coarse | `13/40` |

Thus the historical strict inequality was contradictory: `11/40 == 11/40`.
V2 changes only the rich registered policy to always action 0. The prior,
utility table, coarse/rich channels, garbling witness, risk functional, sample
size, theorem targets, and criterion remain unchanged. The resulting reversal
is attributed only to restricted/non-nested policy classes.

## S9 Full-Information Law

V2 defines strata `L/H` with masses `4/5` and `1/5` and a symmetric
Rademacher variable `R`. One query reveals:

```text
loss_0 = 3/10
loss_1 = 3/10 + 1/20 + sigma_h R
loss_2 = 13/20
loss_3 = 17/20

sigma_L = 1/50
sigma_H = 1/5
```

All four stratum/sign support vectors lie in `[0,1]`. Exact population means
are `(3/10, 7/20, 13/20, 17/20)`, so action 0 is uniquely optimal. The
`loss_1-loss_0` mean is `1/20`, with exact within-stratum SDs `1/50` and
`1/5`.

For budget 64, deterministic largest-remainder allocation gives:

```text
passive proportional:
  n_L / n_H = 51 / 13

Neyman p_h sigma_h:
  n_L / n_H = 18 / 46
```

Using `sum_h p_h^2 sigma_h^2/n_h`:

```text
passive variance:
  1327/10359375 = 0.00012809653092006034

Neyman variance:
  317/6468750 = 0.000049004830917874396
```

The second is lower for this fixed scenario only. C85R makes no universal
active-testing superiority claim and authorizes no acquisition execution.

## S6/S7 Error Law

Both scenarios now use:

```text
u_hat_a = u_a + xi_a
xi_a iid Normal(0, pairwise_sigma^2/2)
selection = argmax u_hat, first canonical index on an exact tie
```

With `pairwise_sigma=1/50`, action-error variance is `1/5000` and
`Var(xi_i-xi_star)=1/2500`. Two pairwise differences sharing `xi_star` have
covariance `1/5000` and correlation `1/2`; they are not independent. The eight
required outputs per scenario are locked for future C85T, but all have
`computed_in_C85R=0`. The 4,096-replicate scientific simulations did not run.

## T7 And Proof Precision

The primary open T7 target is now:

```text
P(selected outside A_epsilon)
  <= min(1, sum_{i:Delta_i>epsilon} exp[-Delta_i^2/(2 sigma_i^2)])
```

It follows the registered implication that selecting action `i` requires
`xi_i-xi_star >= Delta_i`. The historical `(Delta_i-epsilon)^2` expression is
retained as a looser candidate diagnostic. Neither expression is labelled
proved, and no independence assumption is made.

The additive proof-obligation precision also requires:

- T3: statewise almost-sure equality of randomized action kernels, not equality
  of one coupled draw.
- T4: unique different optima or disjoint optimal-action sets with a decoder,
  equal prior, common observation space, nonoptimal regret at least `Delta`,
  randomized rules, and a fixed TV convention.
- T6: derivation of the exact S5 CVaR region strictly inside `alpha in (0,1)`.

T1-T7 all remain `OPEN`.

## Semantic Validation

Only exact contract-preflight operations ran:

```text
S10 exact finite risk enumeration;
S9 support, moment, allocation, and variance identities;
S6/S7 covariance and pairwise-scale identities;
S0-S10 schema and deterministic-seed replay;
T1-T7 OPEN-status replay.
```

The exact status is:

```text
SEMANTICALLY_VALIDATED_NOT_SCIENTIFICALLY_EXECUTED
```

No random draw was generated by the semantic preflight. The SHA-256 seed rule
is replayed for all scenarios without invoking a random-number generator.

## Artifact Inventory

```text
repair protocol/timing/sidecar:       3 files
operative V2 contract/sidecar:        2 files
materialized semantic CSV contracts: 15 files / 92 rows
semantic validator:                   1 module
C85R tests:                            2 files / 30 tests
formal regression wrapper:            1 file
red-team checks:                      64 / 64 PASS
```

The C85R V2 contract, semantic tables, validator, and tests total about 75 KiB.
No raw EEG, label view, candidate array, checkpoint, synthetic scientific
result, weight, cache, or optimizer state is tracked.

## Regression Verification

At implementation commit `360c422f3110d61cfef5d09fead78562bb52c497` in
Python 3.13.7, GPU 0:

```text
focused: 321 passed
C65:     932 passed, 1 skipped, 3 deselected
C23:   1,343 passed, 1 skipped, 3 deselected
full:  2,267 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. `squeue` showed zero active C84/C85/OACI
jobs. `sacct` was not used.

## Protected Boundary

```text
S0-S10 4096-replicate simulations: 0
C85 project proofs completed:      0
theorem-status transitions:        0
EEG/direct-label access:            0 / 0
candidate-array/selector access:    0 / 0
empirical inference or p-values:    0
training/forward/GPU:               0 / 0 / 0
active acquisition execution:       0
C85T/C85E authorization:            0 / 0
new data/model zoo:                  0
manuscript work:                     0
```

## Next Boundary

C85T may begin only after separate PM review and authorization. It must use the
exact V2 contract and may then attempt self-contained proofs, exhaustive
counterexamples, and S0-S10 V2 synthetic execution. C85E, active acquisition,
new data/model zoos, and manuscript work remain unauthorized.

