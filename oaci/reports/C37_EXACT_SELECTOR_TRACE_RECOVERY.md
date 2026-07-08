# C37 - Exact Selector Trace Recovery / Leakage-UCL Audit (frozen C19 `664007686afb520f`)

> Read-only exact replay from C10/C8 Phase-A source-train feature stores and frozen selection bootstrap plans. No training, no re-inference, no proxy selector score, no selected-checkpoint method artifact.

- **cases: `T1_exact_ucl_prefers_selected, T5_selection_audit_inversion_confirmed, T6_source_pareto_conflict_survives_exact_trace, T8_actual_selector_misdirection_supported`**
- P0 selected-UCL identity: **3/3**.
- recovered unique better-candidate UCLs: **38/38**.

## Exact UCL Direction

- UCL prefers selected / better / flat: **114 / 0 / 0** of 114.
- point-vs-UCL disagreement fraction: **0.000**.
- UCL plateau fraction at eps 0.02: **0.132**.

## Reconciliation

- exact selection-UCL to audit inversion rate: **0.447**.
- exact selection-UCL target conflict rate: **1.000**.
- source-Pareto conflict after UCL recovery: **1.000**.

## Boundaries

- Exact UCL replay uses persisted Phase-A source-train features and frozen bootstrap plans; it does not use leakage point, audit leakage, source score, or target endpoints as a selector proxy.
- Ordering is pairwise selected-vs-C35-better, not a full trajectory selector rerank unless all local candidate UCLs are recovered.

## Bottom Line

> C37 closes C36's better-candidate UCL gap when P0 identity and exact replay pass; any T8 claim is conditioned on exact UCL, not point leakage.
