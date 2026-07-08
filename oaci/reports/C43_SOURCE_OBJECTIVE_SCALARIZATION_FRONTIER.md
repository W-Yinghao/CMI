# C43 - Source-Objective Scalarization Frontier Audit (frozen C19 `664007686afb520f`)

> Read-only hindsight diagnostic over a frozen source-only objective registry and fixed scalarization grid. No training, no GPU, no feature selection, no score tuning, and no selected-checkpoint artifact.

- **cases: `F1_source_objective_frontier_contains_target_good, F3_leakage_extreme_blocks_rank_frontier, F4_no_source_scalarization_reliable_topk, F5_hindsight_scalarization_ceiling_weak, F7_source_rank_leakage_tradeoff_real, F8_source_only_scalarization_escape_hatch_closed`**
- candidate rows / trajectories: **3804 / 162**.
- fixed scalarizations: **103** at step **0.10**.

## Best Hindsight Source Scalarization

- best id: **`leakage_rank_risk__leakage_0.5__rank_0.1__risk_0.4`**.
- best top1 joint-good: **0.574** vs trajectory random **0.430**.
- gain vs random: **0.144**.
- Holm p / BH q: **0.000 / 0.000**.
- per-target AUC sign consistency: **1.000**.

## Source Pareto Frontier

- source-front fraction: **0.972**.
- joint-good front fraction: **0.988**.
- Pareto-good front fraction: **0.996**.
- preference-robust target-better front fraction: **0.965**.

## Leakage-Rank Frontier

- mean leakage/rank Spearman: **-0.292**.
- leakage-blocks-rank-better fraction: **0.537**.
- OACI mean leakage percentile: **0.016**.
- OACI mean source-rank percentile: **0.685**.

## Bottom Line

> C43 closes the broader source-only scalarization escape hatch under current artifacts: source objectives contain weak target-relevant signal and the best fixed hindsight mixture improves over random, but the top1/top-k localization remains below reliability gates after multiplicity and stability checks.
