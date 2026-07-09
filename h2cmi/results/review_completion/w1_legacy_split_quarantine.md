# W1 Legacy Split Quarantine

- status: PASS
- original REVIEW_P0 W1 MI results: `legacy_split_not_confirmatory`
- SPDIM P6 seed-0 W1 results: `legacy_split_not_confirmatory`
- Cho2017 old split rows: `single_class_eval_affected`
- BNCI2014_001 old split rows: `metric_valid_under_class_composition`
- Lee2019_MI old split rows: `metric_valid_under_class_composition`

## Required Fields

- affected_dataset: `Cho2017`
- affected_target_subjects: `52`
- affected_fraction: `1.0`
- old_w1_confirmatory_status: `False`
- old_spdim_seed0_baseline_status: `False`
- allowed_future_use: `diagnostic_legacy_only`
- prohibited_future_use: `confirmatory_mi_aggregate_or_spdim_baseline`

## Rationale

P6.1 confirmed Cho2017 has 52/52 single-class evaluation targets under the old W1 split. The project scorer is numerically defined but degenerates to ordinary accuracy on one-class `y_true`; therefore the old W1 MI aggregate and SPDIM P6 seed-0 rows are retained only as diagnostic legacy artifacts.

## Red Team Review

- This quarantine does not delete old artifacts.
- It blocks confirmatory MI aggregate or SPDIM baseline use.
- It does not approve reruns or extra seeds.
