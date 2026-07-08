# C48 - Conditioned Source-Space Ceiling / Local Bayes Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic ceiling audit over fixed source spaces and fixed conditioning groups. Local purity excludes self labels and uses same-group random baselines.

- **cases: `LC1_conditioned_source_space_ceiling_high, LC3_existing_scores_underuse_source_space`**
- candidate rows / trajectories: **3804 / 162**.
- C47 strict-source conditioned top1: hit **0.556**, enrichment **1.307**.

## Local Ceiling

- best conditioned scope: **within_target**.
- best conditioned source space / neighborhood: **all_source_objectives / eps_q02**.
- best conditioned top1 hit / enrichment: **1.000 / 2.360**.
- permutation-adjusted top1 gap: **0.574**.
- gap vs C47 actual strict-source top1: **0.556**.
- mean local purity / base rate at that ceiling: **0.425 / 0.424**.
- mean neighbor count / empty-neighborhood fraction: **1.275 / 0.655**.
- global best top1 / enrichment: **1.000 / 2.357**.
- within-regime best top1 / enrichment: **1.000 / 2.357**.

## Bottom Line

> C48 separates local source-space ceiling from existing source-score actionability. The high ceiling is a sparse max-local diagnostic ceiling, not a broad purity shift and not an action rule. Under the current artifacts, existing source scores underuse that local information.
