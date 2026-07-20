# C42 - Source-Rank Actionability / Rank-to-Selector Gap Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic counterfactual. No training, no GPU, no score tuning, no feature selection, no selected-checkpoint artifact, and no deployable selector claim.

- **cases: `R1_source_rank_pairwise_signal_real, R2_rank_to_topk_gap, R3_source_rank_top1_improves_over_oaci_but_not_reliable, R6_dense_base_rate_limits_claim, R7_top_region_plateau_or_instability, R8_leakage_blocks_rank_better_candidates, R9_source_rank_escape_hatch_closed`**
- candidate rows / trajectories: **3804 / 162**.

## Pairwise Signal

- C30 source-rank AUC from C30: **0.659**.
- C42 source-rank AUC vs C41 continuous target utility: **0.590**.
- C41 selection-leakage AUC: **0.494**.

## Top-1 Actionability

- source-rank top1 joint-good: **0.506**.
- actual OACI top1 joint-good: **0.444**.
- trajectory-conditioned random baseline: **0.430**.
- source-rank top1 regret vs target oracle: **0.657**.
- source-rank top1 target-better-than-OACI fraction: **0.537**.

## Stability And Conflict

- source-rank mean plateau size at eps 0.02: **2.191**.
- low top1/top2 margin fraction: **0.537**.
- leakage-blocks-rank-better fraction: **0.537**.
- Target-gauge delta for source-rank top1 vs OACI is not available as a candidate-level field; no proxy is used.

## Bottom Line

> C42 closes the source-rank escape hatch for deployment/reliable selection: the rank signal is real and often target-better than OACI, but its top1/top-k localization is modest over the high trajectory-conditioned base rate, top regions are plateaued, and regret remains large.
