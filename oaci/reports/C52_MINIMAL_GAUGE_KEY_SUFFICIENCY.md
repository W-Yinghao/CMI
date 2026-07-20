# C52 - Minimal Gauge-Key / Conditioning-Sufficiency Audit (frozen C19 `664007686afb520f`)

## Decision

`C52-G_diagnostic_label_content_required`

## C51 Residual Replay

- C51 decision: `C51-E_target_trajectory_gauge_residual`
- best strict source hit: **0.506**
- C51 trajectory local-Bayes hit: **0.809**
- trajectory-centered diagnostic hit: **0.813**
- N2/N3 fail-fraction percentiles: **0.000 / 0.000**
- N4 enrichment null mean vs observed: **0.834 / 2.360**

## Gauge-Key Ladder

- trajectory random-tie hit: **0.430**
- best key-only hit: **0.488**
- best label-derived diagnostic hit: **0.813**
- source-observable closes gap: **False**
- target / trajectory / target×trajectory key-only closes gap: **False / False / False**
- label-derived diagnostic closes gap: **True**

## Bottom Line

C52 separates key availability from label-derived diagnostic content. Target and trajectory keys are useful as grouping labels for the audit, but key-only baselines remain at the trajectory random-tie level under the frozen within-trajectory evaluation. The C51 residual is closed only when target labels are used diagnostically inside trajectory cells, so the C49-C51 ceiling remains a diagnostic boundary rather than a source-measurement localization result.

## Red-Team Checks

- c51_residual_replayed: PASS - C52 reuses C51 source-score and trajectory diagnostic gaps without rerunning earlier audits.
- key_only_vs_label_derived_not_conflated: PASS - Trajectory key-only remains at the trajectory random-tie baseline while label-derived diagnostics close.
- source_observable_not_sufficient: PASS - Existing source scores and observable source geometry do not reach the closure gate.
- target_key_not_sufficient: PASS - Target id alone has no within-trajectory rank under the frozen evaluation scope.
- trajectory_key_not_sufficient: PASS - Trajectory id alone ties the candidate set and does not explain the residual.
- target_unlabeled_geometry_not_claimed: PASS - No target-unlabeled geometry sufficiency claim is made from unavailable committed artifacts.
- n5_n7_nulls_or_quarantine_emitted: PASS - N5/N6 key nulls and N7 diagnostic quarantine rows are emitted.
- no_selection_artifact: PASS - C52 writes audit ledgers only and does not emit selected-candidate fields.
