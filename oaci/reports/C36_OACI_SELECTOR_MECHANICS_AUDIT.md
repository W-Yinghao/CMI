# C36 - OACI Selector Mechanics / Feasibility-Regret Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic audit over C35/C34S/C10 artifacts. C36 asks what the actual OACI selector trace contains for C35 preference-robust local better alternatives. No training, no re-inference, no selector, no selected-checkpoint artifact.

- **cases: `S2_leakage_objective_prefers_selected, S4_selection_audit_inversion, S5_source_pareto_conflict, S9_trace_insufficient`**
- preference-robust C35 pair-rows: **114** over **38** unique selector units; C35's frozen raw utility grid is unchanged (step **0.05**).

## Stage-0 Trace Availability

- selected UCL available for selected rows: **114/114**.
- better-candidate UCL available: **0/114**.
- Therefore C36 does not compute a numeric actual-selector score delta or actual UCL plateau. Selection-leakage point is reported only as a component trace, not as a proxy selector score.

## Feasibility And Leakage Components

- risk-gate regret fraction: **0.000**.
- selection-leakage point component prefers selected: **1.000**.
- source endpoint majority prefers selected: **0.474**.
- trace-unavailable fraction for actual selector deltas: **1.000**.

## Selection-Audit Inversion

- selection-to-audit inversion rate: **0.447**.
- audit-to-target inversion rate: **0.553**.
- local leakage-target conflict rate: **1.000**.

## Source Pareto

- source-Pareto conflict fraction (selected dominates or incomparable): **1.000**.
- better source-dominates selected fraction: **0.000**.
- The source-Pareto objective set is frozen before analysis and uses source risk, leakage point components, source_guard endpoints, and source_audit endpoints; it is not a scalar selector. Non-finite endpoint cells are filtered per pair and the finite objective count is reported in the table.

## Plateau / Tie

- actual selector UCL plateau available: **False**.
- point-component active selected-margin fraction at eps 0.02: **0.895**.
- S6/S7 are blocked as actual-selector claims because better-candidate UCLs and tie metadata are absent.

## Bottom Line

> C36 localizes the C35 robust local misses to a source-side selector trace where risk-feasible better alternatives are present, the selection-leakage point component consistently favors the artifact-selected candidate, source-Pareto conflict is common, and exact per-candidate UCL/tie trace is insufficient for numeric actual-selector margin claims.
