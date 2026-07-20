# C49 - Sparse Local-Bayes Coverage Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic audit over fixed local-Bayes neighborhoods. Coverage, empty-neighborhood fraction, and same-group random baselines are reported with every curve point.

- **cases: `SC2_broad_conditioned_ceiling, SC3_existing_scores_underuse_available_islands, SC5_ceiling_unstable_across_targets, SC8_conditioned_source_space_future_hypothesis`**
- candidate rows / trajectories: **3804 / 162**.
- C48 ceiling reference: hit **1.000**, enrichment **2.360**, permutation gap **0.574**.

## Coverage Gate

- best conditioned setup: **within_target / all_source_objectives / eps_q01 / min_n=5**.
- best hit / enrichment / coverage: **1.000 / 2.656 / 0.023**.
- coverage >= 0.50 reliable: **True** via **within_target / all_source_objectives / eps_q20 / min_n=1** (hit **1.000**, coverage **1.000**).
- coverage >= 0.75 reliable: **True** via **within_target / all_source_objectives / eps_q20 / min_n=1** (hit **1.000**, coverage **1.000**).

## Stability And Underuse

- target min hit / min coverage: **1.000 / 0.000**.
- trajectory min hit / min coverage: **0.000 / 0.000**.
- max existing-score underuse gap: **C19_robust_core = 1.000**.

## Bottom Line

> C49 tests whether C48's sparse max-local ceiling survives coverage and stability gates. The result is diagnostic-only: any local ceiling still uses target labels for scoring and does not create a deployable action rule.
