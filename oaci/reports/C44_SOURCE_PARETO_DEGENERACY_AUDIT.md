# C44 - Source-Pareto Degeneracy Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic audit explaining C43's broad source Pareto frontier. No training, no GPU, no feature selection, no re-inference, and no selected-checkpoint artifact.

- **cases: `PF1_source_pareto_front_degenerate, PF2_front_membership_non_discriminative, PF3_objective_conflict_inflates_front, PF4_objective_redundancy_not_the_issue, PF5_family_reduced_frontier_narrows_but_loses_target_coverage, PF6_family_reduced_frontier_has_diagnostic_signal, PF7_dominance_depth_not_target_informative, PF9_source_objective_geometry_non_identifiable`**
- candidate rows / trajectories: **3804 / 162**.

## Pareto Degeneracy

- observed source-front fraction: **0.972**.
- Gaussian same-dimension null front fraction: **0.981**.
- objective-shuffled null front fraction: **0.981**.

## Objective Geometry

- effective rank: **5.696**.
- first-PC variance: **0.414**.
- negative objective-pair fraction: **0.444**.
- leakage/rank family mean Spearman: **-0.071**.

## Front Membership

- P(joint-good | front): **0.431**.
- trajectory baseline joint-good: **0.430**.
- P(joint-good | not-front): **0.529**.
- dominance-depth AUC vs target utility: **0.499**.

## Bottom Line

> C44 explains C43's F1 caveat: source-front membership is broad and non-discriminative. The source objective geometry contains conflict and high effective dimension, so target-good-on-front does not imply source-side identifiability.
