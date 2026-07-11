# W1 Split/Metric Impact Verdict

> **HISTORICAL LEGACY-SPLIT VERDICT:** These fields describe the P6
> contiguous-split checkpoint. P7 repaired the split, and P9 completed the
> official repaired-split SPDIM baseline. Current status is in
> `FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md`.

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

At the P6 checkpoint, Cho2017 had single-class evaluation for all 52 W1
targets under `contiguous_split`; `balanced_accuracy_score` ignored absent
classes and degenerated to ordinary accuracy on those rows. The then-current
W1 geometry signal was therefore affected, and SPDIM seeds 1/2 were correctly
blocked until the repaired protocol was approved.

## Decision Rules Applied

- Cho2017 single-class fraction is substantial: 52/52.
- The project scorer ignores absent classes rather than assigning absent-class recall zero or one.
- REVIEW_P0 reports the W1 geometry effect as Cho2017-driven.
- Full SPDIM expansion was blocked at this historical checkpoint; P9 later
  resolved that blocker under the repaired split.

## Red Team Review

- This verdict does not approve seeds 1/2 or a full three-seed SPDIM run.
- This verdict does not edit manuscript TeX.
- This verdict treats P6 as a seed-0 same-split expansion with a serious Cho2017 split caveat, not a full baseline.
