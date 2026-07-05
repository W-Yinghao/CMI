# C17 — Identifiability case taxonomy

- **CASE: `case_III_multivariate_weak_identifiability`**

## Inputs

- univariate_verdict: weak_accuracy_needs_multivariate
- n_strong_accuracy_signals: 0
- n_weak_accuracy_signals: 3
- n_signals_identify_nll: 5
- oracle_signal_spearman_bacc: 0.119523893328065
- max_abs_accuracy_spearman: 0.23603865667990312
- accuracy_signal_families: ['accuracy', 'objective', 'risk']
- multivariate_loto_auc: 0.6023104389834069
- multivariate_beats_permutation: True
- source_signals_calibration_biased: True

## Interpretation
> no scalar source signal works, but source-only combinations weakly beat permutation -> competence information exists but is not captured by simple selectors

## Next science
> a low-freedom source-only competence probe MAY be pre-registered; not deployable yet