# Step 15 — coverage-aware harm-control policies

Scope: coverage-aware harm-control policies (R2 minimal-label); not SOTA. R2 labeled-slice policy under an iid sampling contract; oracle full gain used only for evaluation; NOT R1 target-gain identifiability; oracle_full_target is not deployable.

- runs: **54** · policies **7** · ks **[0, 1, 2, 4, 8, 16, 32, 64, 128, 256]** · taus **[0.0, 0.01, 0.02, 0.05]** · repeats **500** · CI **normal**
- always-adapt harm-rate: **0.8519** · harm constraint **0.05**
- best deployable policy: **None** — no deployable policy meets the harm<=0.05 constraint while adapting (adaptation_coverage>0)

| policy | k | tau | adapt_cov | decision_cov | abstain | harm@adapt | prevented_harm | missed_benefit | cond_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_identity | 16 | 0.0 | 0.0 | 1.0 | 0.0 | None | 1.0 | 1.0 | 0.8519 |
| always_adapt | 16 | 0.0 | 1.0 | 1.0 | 0.0 | 0.8519 | 0.0 | 0.0 | 0.1481 |
| plugin_sign | 16 | 0.0 | 0.2554 | 1.0 | 0.0 | 0.7807 | 0.7659 | 0.622 | 0.7084 |
| ci_adapt_only_abstain | 16 | 0.0 | 0.0078 | 0.0078 | 0.9922 | 0.8246 | 0.9924 | 0.9908 | 0.1754 |
| ci_adapt_only_identity | 16 | 0.0 | 0.0078 | 1.0 | 0.0 | 0.8246 | 0.9924 | 0.9908 | 0.8468 |
| ci_three_way | 16 | 0.0 | 0.0078 | 0.0633 | 0.9367 | 0.8246 | 0.9924 | 0.9908 | 0.8824 |
| oracle_full_target | 16 | 0.0 | 0.1481 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |

> Table shows k=16, tau=0.0; full grid in the JSON.
> R2 labeled-slice policy under an iid sampling contract; oracle full gain used only for evaluation; NOT R1 target-gain identifiability; oracle_full_target is not deployable.
