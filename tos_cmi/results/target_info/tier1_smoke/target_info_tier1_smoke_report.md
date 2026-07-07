# Fork 1 Tier-1 smoke --- target-information budget curves (semi-synthetic; NOT a final paper claim)

Scope: ['Lee2019_MI', 'Cho2017'] x ['EEGNet'] x folds [1, 2, 3, 4, 5] x worlds ['v2_source_invisible_world_a', 'source_rich_source_visible_world_a'] x budgets ['B0_source_only', 'B1_unlabeled_target', 'B2_k_labels_per_class', 'B3_sequential_calibration', 'B4_oracle_selector'] ; k=[1, 2, 4, 8, 16] ; R=10 ; alpha=[0.5, 1.0, 2.0] ; split=subject_seeded_v1 ; n_boot=100.
Decision rows 26460 ; audit rows 26460 ; workers 420 ; failures 0.

## Stop-condition audit (all must be 0)
```
b1_accepts                 0
b4_deployable_accepts      0
unflagged_non_specific     0
point_estimate_safety      0
```

## Per-budget action counts (deployable B0/B1/B2/B3 ; diagnostic B4)
- B0_source_only             {'abstain': 318, 'reject': 102}
- B1_unlabeled_target        {'abstain': 318, 'reject': 102}
- B2_k_labels_per_class      {'abstain': 15580, 'reject': 5100, 'accept': 320}
- B3_sequential_calibration  {'abstain': 3164, 'reject': 1020, 'accept': 16}
- B4_oracle_selector         {'DIAGNOSTIC': 420}

## B2 k-curve (per world): accept rate, true/false accept, held-out audit ΔbAcc, specificity
| world | k | n | accept_rate | true_acc | false_acc | mean_audit_ΔbAcc | specific | non_specific |
|---|---|---|---|---|---|---|---|---|
| source_rich_source_visible | 1 | 2100 | 0.00 | 0 | 3 | -0.010 | 3 | 0 |
| source_rich_source_visible | 2 | 2100 | 0.01 | 7 | 5 | 0.008 | 12 | 0 |
| source_rich_source_visible | 4 | 2100 | 0.01 | 9 | 8 | 0.008 | 18 | 0 |
| source_rich_source_visible | 8 | 2100 | 0.01 | 20 | 7 | 0.018 | 28 | 0 |
| source_rich_source_visible | 16 | 2100 | 0.02 | 24 | 12 | 0.017 | 37 | 0 |
| v2_source_invisible_world_ | 1 | 2100 | 0.01 | 0 | 13 | -0.010 | 13 | 0 |
| v2_source_invisible_world_ | 2 | 2100 | 0.01 | 22 | 2 | 0.035 | 24 | 0 |
| v2_source_invisible_world_ | 4 | 2100 | 0.02 | 28 | 7 | 0.034 | 35 | 0 |
| v2_source_invisible_world_ | 8 | 2100 | 0.03 | 57 | 5 | 0.041 | 62 | 0 |
| v2_source_invisible_world_ | 16 | 2100 | 0.04 | 77 | 11 | 0.037 | 88 | 0 |

## B3 sequential calibration
- actions: {'abstain': 3164, 'reject': 1020, 'accept': 16}
- mean label budget used before decision (k_used): 1.0

## B4 oracle diagnostic (upper bound; excluded from deployable accept counts)
- oracle audit ΔbAcc mean over 0 cells: None

Budget curve: `tos_cmi/results/target_info/tier1_smoke/target_info_tier1_budget_curve.png`

## Reading guide
- B0 source-only expected to abstain/reject on source-invisible benefit; B1 must NEVER accept (stop-cond).
- B2/B3 accept is the target-information signal: SAFE only if held-out audit ΔbAcc > +0.01 AND same-k random does not reproduce it (accepted_specific). accepted_non_specific / false_accept are disclosed.
- Many abstains at small k are EXPECTED (weak calibration LCB), not a failure.

