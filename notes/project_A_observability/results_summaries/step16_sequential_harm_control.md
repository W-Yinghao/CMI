# Step 16 — sequential label-acquisition harm-control

Scope: sequential label-acquisition harm-control (R2); not SOTA. R2 sequential labeled-slice policy under an iid sampling contract; NOT R1 target-gain identifiability; oracle_full_target is an evaluation-only upper bound, not deployable.

- runs: **54** · budgets **['8', '16', '32', '64', '128', '256', '512', 'full']** · batch **8** · repeats **500** · always-adapt harm **0.8519**
- best sequential deployable policy: **None** — no sequential policy meets harm<=0.05 and coverage>=0.05
- oracle reference (eval-only): adapt-cov 0.1481, harm 0.0

| policy | budget | tau | adapt_cov | mean_labels | harm@adapt | meets<=0.05 | missed_benefit |
|---|---|---:|---:|---:|---:|---|---:|
| seq_ci_three_way | 8 | 0.0 | 0.0113 | 8.0 | 0.8007 | False | 0.9848 |
| seq_ci_three_way | 16 | 0.0 | 0.0161 | 15.54 | 0.7747 | False | 0.9755 |
| seq_ci_three_way | 32 | 0.0 | 0.0279 | 29.61 | 0.7301 | False | 0.9493 |
| seq_ci_three_way | 64 | 0.0 | 0.038 | 54.22 | 0.6693 | False | 0.9153 |
| seq_ci_three_way | 128 | 0.0 | 0.047 | 96.23 | 0.615 | False | 0.8778 |
| seq_ci_three_way | 256 | 0.0 | 0.0536 | 164.91 | 0.5622 | False | 0.8417 |
| seq_ci_three_way | 512 | 0.0 | 0.057 | 269.66 | 0.5325 | False | 0.82 |
| seq_ci_three_way | full | 0.0 | 0.0574 | 314.26 | 0.529 | False | 0.8175 |
| seq_ci_adapt_only | 8 | 0.0 | 0.0113 | 8.0 | 0.8007 | False | 0.9848 |
| seq_ci_adapt_only | 16 | 0.0 | 0.0161 | 15.91 | 0.7747 | False | 0.9755 |
| seq_ci_adapt_only | 32 | 0.0 | 0.0279 | 31.6 | 0.7301 | False | 0.9493 |
| seq_ci_adapt_only | 64 | 0.0 | 0.038 | 62.57 | 0.6696 | False | 0.9153 |
| seq_ci_adapt_only | 128 | 0.0 | 0.0471 | 123.85 | 0.6153 | False | 0.8778 |
| seq_ci_adapt_only | 256 | 0.0 | 0.0536 | 245.39 | 0.5628 | False | 0.8417 |
| seq_ci_adapt_only | 512 | 0.0 | 0.0573 | 487.07 | 0.5317 | False | 0.819 |
| seq_ci_adapt_only | full | 0.0 | 0.0576 | 617.22 | 0.5283 | False | 0.8165 |
| seq_plugin_confirm | 8 | 0.0 | 0.0 | 8.0 | None | False | 1.0 |
| seq_plugin_confirm | 16 | 0.0 | 0.1564 | 16.0 | 0.7842 | False | 0.7722 |
| seq_plugin_confirm | 32 | 0.0 | 0.2607 | 22.29 | 0.7668 | False | 0.5897 |
| seq_plugin_confirm | 64 | 0.0 | 0.3119 | 25.27 | 0.7557 | False | 0.4858 |
| seq_plugin_confirm | 128 | 0.0 | 0.321 | 25.82 | 0.7495 | False | 0.4572 |
| seq_plugin_confirm | 256 | 0.0 | 0.3214 | 25.85 | 0.7491 | False | 0.4557 |
| seq_plugin_confirm | 512 | 0.0 | 0.3214 | 25.85 | 0.7491 | False | 0.4557 |
| seq_plugin_confirm | full | 0.0 | 0.3214 | 25.85 | 0.7491 | False | 0.4557 |

> Table shows tau=0.0; full grid in the JSON.
> R2 sequential labeled-slice policy under an iid sampling contract; NOT R1 target-gain identifiability; oracle_full_target is an evaluation-only upper bound, not deployable.
