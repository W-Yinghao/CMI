# C32 - Joint-Good Localization / Selection-Regret Anatomy Audit (frozen C19 `664007686afb520f`)

> C31 showed that joint accuracy+calibration-good checkpoints are common and that the C16 barrier is a source-observability / gauge / localization failure, not a checkpoint-space trade-off. C32 asks why source-side selection and diagnostic scores still do not localize the common joint-good set. Read-only over C10 + C22 + C24; diagnostic-only; no selector, no training, no selected-checkpoint artifact.

- **cases: `J1_joint_good_common_not_scarce, J2_trajectory_random_baseline_nontrivial, J3_source_topk_localization_weak, J4_selection_regret_is_localization_not_scarcity, J5_selected_oaci_near_joint_good, J6_target_unlabeled_pooling_help_no_topk_rescue, J7_target_grouped_rank_recovers_pooled_localization_non_deployable`**

## Gate 1 - joint-good landscape (scarcity check)

- candidates: **3804** across **162** trajectory-regime units.
- joint-good rate: **42.4%** (1614 candidates); trajectory-regime units with at least one joint-good: **94.4%**.
- mean / median joint-good per trajectory-regime unit: **+9.963 / +8.500**; min/max: **0 / 29**.

## Gate 2 - trajectory-conditioned random baseline

| k | random hit | source-score hit | enrichment |
|---:|---:|---:|---:|
| 1 | 43.0% | 50.6% | +1.178 |
| 3 | 65.8% | 67.3% | +1.023 |
| 5 | 75.0% | 81.5% | +1.086 |
| 10 | 85.2% | 88.9% | +1.043 |

- random top-1 is already **43.0%** because joint-good is common.
- source-score top-1 improves to **50.6%** (enrichment +1.178), but top-3/top-5 are only weakly above the trajectory-conditioned random baseline: **67.3% / 81.5%** vs **65.8% / 75.0%** (enrichment +1.023 / +1.086).

## Gate 3 - selected OACI regret anatomy

- selected OACI joint-good hit: **44.4%**, essentially random top-1 (43.0%); scarcity/no-joint trajectories are only **5.6%**.
- nearest joint-good distance from selected OACI: median order **+1.000**, mean order **+2.588**; median epoch **+5.000**.
- selected-to-best-joint regret: bAcc mean **+0.055** (median +0.052, max +0.182); NLL/ECE mean regret **+0.150 / +0.055**.

| selected-regret category | fraction | count |
|---|---:|---:|
| adjacent_near_miss | 13.0% | 21 |
| scarcity_no_joint_good | 5.6% | 9 |
| selected_joint_good | 44.4% | 72 |
| source_rank_not_enriched_enough | 12.3% | 20 |
| source_top5_available_selection_missed | 24.7% | 40 |

## Gate 4 - source-only / target-unlabeled / target-grouped localization ladder

| information rung | pooled AUC | within-target AUC | top-1 hit | top-5 hit |
|---|---:|---:|---:|---:|
| source_score | +0.541 | +0.672 | 50.6% | 81.5% |
| target_unlabeled_loto | +0.583 | +0.562 | 35.2% | 75.9% |
| source_plus_target_unlabeled_loto | +0.569 | +0.585 | 40.1% | 79.0% |
| target_grouped_centered_score | +0.645 | +0.672 | 50.6% | 81.5% |

- target-unlabeled confidence geometry improves **pooled** localization by **+0.042** AUC over source score, but its top-1 trajectory localization is **-0.154** relative to source. This is a weak pooling/gauge aid, not a top-k rescue.
- target-grouped centering improves pooled AUC by **+0.105** and recovers the within-target rank signal, but it uses target grouping and is non-deployable.

## Margin sensitivity (robust margin 0.02)

- robust joint-good rate **27.8%**; trajectories with joint-good **77.8%**; selected hit **29.6%**; cases **J2_trajectory_random_baseline_nontrivial, J3_source_topk_localization_weak, J6_target_unlabeled_pooling_help_no_topk_rescue, J7_target_grouped_rank_recovers_pooled_localization_non_deployable**.

## Bottom line

> Joint-good checkpoints are common, and selected OACI is usually close to one, but the source-side localization signal is too weak and gauge-broken to choose them reliably. Random top-1 is already high because the set is common; source score gives only mild top-1 enrichment and only weak top-k enrichment; selected OACI lands near random and most regret is non-scarcity localization regret. Target-unlabeled features help the pooled gauge weakly but do not rescue top-k localization. Target grouping largely repairs the pooled rank/gauge mismatch, confirming the C31 reading, but that is an oracle diagnostic, not deployable.