# C18 — support-stress taxonomy

- **CASE: `collapsed_by_accuracy_endpoint_nonestimability`**

## Evidence

- s0_beats_permutation: True
- regime_collapse_reason: {'S0_full_support': 'none', 'S1_label_marginal_skew': 'implemented_noop', 'S2_rare_cells': 'none', 'S3_nonestimable_cells': 'none', 'S4_missing_cells': 'endpoint_metric_nonestimability', 'S5_block_class_by_domain': 'signal_loss', 'S6_boundary_aligned_mask': 'endpoint_metric_nonestimability', 'S7_random_matched_mask': 'endpoint_metric_nonestimability'}
- cell_present_preserved_fraction: 0.6666666666666666
- cell_deletion_endpoint_nonestimability_fraction: 1.0
- deleting_regimes_with_comparable_remaining: 1.0
- deleting_regimes_leakage_estimable_fraction: 1.0
- boundary_s6_corr: 0.5110324718441439
- boundary_s7_corr: 0.46207633725439035
- mean_accuracy_visibility_deleting: 0.07553101992113682
- mean_calibration_visibility_deleting: 0.14214681574553442

## Interpretation
> the weak signal SURVIVES cell-present stress (rare/nonestimable); it collapses only under cell DELETION, and there because the worst-domain accuracy ENDPOINT becomes non-estimable (a domain loses a class -> reference bAcc NaN), not because the model signal vanished. Support deletion destroys accuracy-observable availability before it forces leakage abstention.

## Next science
> the limiter is estimator-level accuracy-endpoint availability under cell deletion, not signal loss; a pre-registered competence probe should use deletion-robust (calibration/leakage) observables + report endpoint estimability. Still diagnostic, NOT deployable.
