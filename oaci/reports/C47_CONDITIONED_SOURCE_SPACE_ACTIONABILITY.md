# C47 - Conditioned Source-Space Actionability Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic audit over committed C46/C45/C43 artifacts. Group conditioning is evaluated against same-group random baselines.

- **cases: `GCA1_conditioning_restores_source_neighborhood_homogeneity, GCA2_conditioning_improves_but_not_reliable_actionability, GCA5_grouped_actionability_still_base_rate_limited, GCA6_global_source_only_comparability_fails, GCA7_group_conditioning_is_separate_problem_class`**
- candidate rows / trajectories: **3804 / 162**.
- inherited C46 q10 divergent rates: within-target **0.005**, within-trajectory **0.133**, cross-target **0.937**.

## Group-Conditioned Top1

- global strict-source best: **R_src**, hit **0.000**, random **0.424**, gain **-0.424**.
- within-target strict-source best: **C19_robust_core**, hit **0.444**, random **0.424**, gain **0.021**.
- within-trajectory strict-source best: **C19_robust_core**, hit **0.506**, random **0.430**, gain **0.076**.
- within-regime strict-source best: **R_src**, hit **0.000**, random **0.424**, gain **-0.424**.

## Smoothing And Sign

- max strict-source primary top1 smoothing gain delta: **0.111**.
- max strict-source pairwise AUC: global **0.520**, within-target **0.597**, within-trajectory **0.601**.

## Bottom Line

> Conditioning preserves the C46 local homogeneity result and improves diagnostic localization in some grouped views, but the strict source fields remain below reliability gates. The actionable object here is a group-conditioned diagnostic problem class, not a target-free method.
