# W1 Split/Metric Impact Verdict

- status: AFFECTED
- cho2017_single_class_eval_confirmed: `True`
- main_w1_cho2017_claim_affected: `True`
- spdim_p6_cho2017_result_affected: `True`
- current_w1_results_can_be_used_as_confirmatory: `False`
- current_spdim_p6_can_be_used_as_seed0_baseline: `False`
- approve_spdim_seeds_1_2: `False`
- require_alternative_w1_split: `True`
- require_metric_recompute: `True`

## Reason

Cho2017 has single-class evaluation for all 52 W1 targets under contiguous_split; sklearn balanced_accuracy_score ignores absent classes and degenerates to ordinary accuracy on those rows. The main W1 geometry signal is Cho2017-driven, so current W1 aggregate claims are affected and SPDIM seeds 1/2 should not launch.

## Decision Rules Applied

- Cho2017 single-class fraction is substantial: 52/52.
- The project scorer ignores absent classes rather than assigning absent-class recall zero or one.
- REVIEW_P0 reports the W1 geometry effect as Cho2017-driven.
- Full SPDIM expansion remains blocked until PM review after this audit.

## Red Team Review

- This verdict does not approve seeds 1/2 or a full three-seed SPDIM run.
- This verdict does not edit manuscript TeX.
- This verdict treats P6 as a seed-0 same-split expansion with a serious Cho2017 split caveat, not a full baseline.
