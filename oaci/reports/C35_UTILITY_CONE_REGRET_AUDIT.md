# C35 - Utility-Cone / Pareto Regret Robustness Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic audit over C34S compact JSON + c34_tables CSVs. C35 asks whether C34 local continuous regret is robust to endpoint preferences or dependent on fixed scalar/norm summaries. No training, no re-inference, no selector, no selected-checkpoint artifact.

- **cases: `U1_preference_robust_local_regret, U2_preference_dependent_tradeoff_regret, U3_pareto_dominated_selected_cases_common, U5_source_active_misranking_preference_robust, U7_target_unlabeled_no_preference_robust_rescue`**

## C34S Gates

- manifest resolves / hashes match / key numbers reconstruct / no legacy monolithic dependency: **{'G0_manifest_resolves': True, 'G1_table_hashes_match': True, 'G2_key_numbers_reconstruct': True, 'G3_no_legacy_monolithic_dependency': True}**.
- reconstructed C34: cases **M2_continuous_source_active_misranking, M7_target_unlabeled_pooled_only_reconfirmed, M8_continuous_endpoint_tradeoff_local**, real-regret **+0.941**, threshold-only **+0.000**.

## Endpoint Vectors First

C34 stores NLL/ECE as improvements, so all endpoint deltas are higher-is-better in C35.
- selected -> alternative mean raw vector from C34: bAcc **+0.020**, NLL-improve **+0.055**, ECE-improve **+0.022**.

## Pareto And Utility-Cone Results

- strict+weak Pareto-better fraction: **+0.471**; incomparable fraction **+0.529**.
- utility-cone robust / dependent / narrow / no-regret fractions: **+0.745 / +0.176 / +0.078 / +0.000**.
- mean weight-simplex fraction where the alternative wins: **+0.814**.
- `preference_robust` means the alternative wins for at least 80% of the frozen nonnegative raw utility grid at step 0.05; it is not a claim over every possible monotone utility.
- U1 and U3 are not the same claim: 72/153 alternatives strictly Pareto-dominate selected, while 81/153 remain Pareto-incomparable tradeoffs. U2 is retained for that tradeoff mass.

## Source And Target-Unlabeled Direction

- robust-case source misranking / agreement: **+0.281 / +0.544** vs random 0.500.
- U5 is read as preference-robust active misranking in a substantial minority of robust cases, not as source scores being mostly backward.
- robust-case target-unlabeled agreement: **+0.342** (non-source-only diagnostic).
- U7 is specifically an R3 local preference-robust non-rescue claim; it is not a general claim that all target-unlabeled geometry fails.

## Scaling Sensitivity

- robust/dependent/narrow fraction ranges across raw, global-z, within-z, rank: **+0.118 / +0.176 / +0.078**.
- U8 is not established because the robust fraction remains the majority under all frozen scalings, even though the exact robust/dependent split moves.
- G0-G3 are artifact-integrity checks over C34S compact JSON and table hashes; no-selector/no-training rows are code-audit assertions for this C35 path, not dynamic call-graph proofs.

## Bottom Line

> C35 separates C34 scalar/norm regret from preference-robust regret. Pareto and utility-cone results determine whether selected OACI is broadly worse across endpoint weights or mostly involved in endpoint tradeoffs. Target-unlabeled and target endpoint quantities remain diagnostic-only and non-source-only.
