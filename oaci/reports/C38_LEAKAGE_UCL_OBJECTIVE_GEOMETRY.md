# C38 - Leakage-UCL Objective Geometry / Source-Target Conflict Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic audit over C37/C36/C35/C34/C27/C29 artifacts. No training, no re-inference, no selector repair, no selected-checkpoint artifact.

- **cases: `L1_point_leakage_drives_ucl_preference, L5_selection_audit_leakage_inversion, L6_source_rational_target_wrong, L7_leakage_target_gauge_conflict, L8_leakage_endpoint_decoupled, L10_trace_atom_insufficient`**
- C37 exact-UCL pairs imported: **114**.

## Stage-0 Availability

- Exact selection point, exact selection UCL, derived bootstrap width, source-audit point, source endpoints, source-Pareto status, and C34 target-gauge delta are available.
- Fold/class/domain/support-cell leakage atom contributions are **not persisted**; C38 therefore reports L10 and makes no class/domain/support atom concentration claim.

## UCL Point/Width Geometry

- UCL prefers selected / better: **114 / 0**.
- Point leakage prefers selected: **114 / 114**.
- Point-dominant rows: **111 / 114**.
- Mean deltas better-selected: point **0.142**, width **-0.032**, UCL **0.110**.

## Selection-To-Audit

- Selection-UCL to audit leakage inversion rate: **0.447**.
- Source-audit leakage prefers selected / better: **63 / 51**.

## Source-Target Conflict

- Source-rational target-wrong fraction: **1**.
- Source endpoint majority selected / better / flat: **54 / 57 / 3**.
- Source-Pareto conflict fraction after C37 exact UCL recovery: **1**.

## Target Gauge

- Target gauge prefers better / selected: **105 / 9**.
- Leakage-vs-target-gauge conflict fraction: **0.921**.

## Support Boundary

- Regime counts: **{'S0_full_support': 38, 'S2_rare_cells': 38, 'S3_nonestimable_cells': 38}**.
- Pair keys invariant across S0/S2/S3: **38 / 38**.

## Bottom Line

> C38 finds that the exact selector-UCL preference is primarily a point-leakage advantage, not a bootstrap-width rescue. That source leakage direction is locally source-rational under C37's recovered source-Pareto geometry but target-wrong for C35 preference-robust alternatives, and it usually opposes the C34 target-gauge direction. Atom-level class/domain/support explanations remain unavailable.
