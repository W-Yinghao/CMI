# C40 - Leakage Point Drift Forensics / Atom-Trace Boundary Closure (frozen C19 `664007686afb520f`)

> Read-only forensic audit over C39/C38/C37 artifacts. No training, no GPU, no selector repair, and no change to the frozen 1e-9 identity gate for elevated atom claims.

- **cases: `D2_numeric_only_drift_bounded, D4_aggregate_vs_atom_path_divergence, D5_diagnostic_atom_pattern_stable_but_blocked, D6_atom_trace_irrecoverable_final, D7_future_instrumentation_required`**
- selection identity at 1e-9: **48 / 76**.
- max absolute drift: **0.000215216**; all selection rows pass at 1e-3: **True**.

## First Divergence

- observed semantic mismatch count: **0**.
- aggregate-vs-atom path divergence count: **28**.
- Feature population, support graph availability, fold plan availability, cell-mass accounting, and atom additive aggregation pass in the committed C39 trace; divergence appears only when the recomputed point is compared to the persisted C37 aggregate point.

## Numeric Boundary

- bounded at 1e-3: **True**.
- positive / negative signed drift rows: **40 / 36**.
- Current artifacts do not persist per-fold probe outputs, so C40 cannot restore exact identity or prove the first sub-stage below aggregate point comparison.

## Diagnostic Stability

- point-direction stable under observed drift: **1**.
- diagnostic broad pattern count: **108 / 114**.
- diagnostic atom-gauge conflict count: **105 / 114**.
- These diagnostic patterns remain blocked; they do not establish atom mechanism.

## Bottom Line

> C40 localizes the C39 failure to persisted aggregate point identity, not atom additivity or an observed support/fold/population mismatch. The drift is bounded in the committed tables, but exact identity is not restored; A9 remains a trace boundary and future atom claims require new persisted per-fold/per-cell trace fields.
