# C46 - Conditioning Boundary Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic audit over C45 source space. Conditioning variables are used only to explain where source-neighborhood meaning holds or breaks.

- **cases: `CB1_source_space_informative_after_target_or_trajectory_conditioning, CB2_cross_target_grouping_breaks_source_equivalence, CB3_within_trajectory_neighborhoods_relatively_homogeneous, CB4_source_only_global_comparability_nonidentifiable, CB6_regime_conditioning_partial_not_sufficient`**
- candidate rows / trajectories: **3804 / 162**.
- inherited source objectives: **17**.

## Boundary

- within-target q10 target-divergent rate: **0.005**.
- within-trajectory q10 target-divergent rate: **0.133**.
- within-regime q10 target-divergent rate: **0.299**.
- cross-target q10 target-divergent rate: **0.937**.
- cross-regime q10 target-divergent rate: **0.004**.

## Variance

- target-conditioned utility variance / global: **0.753**.
- trajectory-conditioned utility variance / global: **0.385**.
- target eta^2 for utility: **0.247**.
- trajectory eta^2 for utility: **0.615**.

## Bottom Line

> C46 interprets C45 as conditioning-sensitive non-identifiability: source neighborhoods are useful inside target/trajectory groupings but lose global comparability across target boundaries. Regime alone is not the break: cross-regime same-target neighborhoods remain homogeneous, while conditioning only on regime still mixes targets and remains ambiguous.
