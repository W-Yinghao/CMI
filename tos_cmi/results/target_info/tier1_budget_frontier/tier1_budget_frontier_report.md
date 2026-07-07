# Fork 1 Tier-1 smoke --- target-information budget curves (semi-synthetic; NOT a final paper claim)

Scope: ['Lee2019_MI', 'Cho2017'] x ['EEGNet'] x folds [1, 2, 3, 4, 5] x worlds ['v2_source_invisible_world_a', 'source_rich_source_visible_world_a'] x budgets ['B0_source_only', 'B1_unlabeled_target', 'B2_k_labels_per_class', 'B3_sequential_calibration', 'B4_oracle_selector'] ; k=[1, 2, 4, 8, 16, 24, 32, 40, 50] ; R=10 ; alpha=[0.5, 1.0, 2.0] ; split=subject_seeded_v1 ; n_boot=100.
Decision rows 43260 ; audit rows 43260 ; workers 420 ; failures 0.

## Stop-condition audit (all must be 0)
```
b1_accepts                 0
b4_deployable_accepts      0
unflagged_non_specific     0
point_estimate_safety      0
b3_k1_accepts              0
```

## Per-budget action counts (deployable B0/B1/B2/B3 ; diagnostic B4)
- B0_source_only             {'abstain': 318, 'reject': 102}
- B1_unlabeled_target        {'abstain': 318, 'reject': 102}
- B2_k_labels_per_class      {'abstain': 28620, 'reject': 9180}
- B3_sequential_calibration  {'abstain': 3180, 'reject': 1020}
- B4_oracle_selector         {'DIAGNOSTIC': 420}

## Deployable (B2+B3) safety summary
- deployable accepts: 0 ; false accepts (audit<=0): 0 ; harmful (audit<-0.01): 0 ; false-accept rate 0.000

## Sample-complexity thresholds (per world)
_cal-LCB shown CLIPPED to the [-1,1] balanced-accuracy-difference range; the raw (unclipped) bound is valid even below -1 and is kept in the summary JSON for diagnostics._
```
source_rich_source_visible_world_a       min_k_true_accept=None  min_k_false<=5%=None  any_accept_at_max_k=False  best_cal_LCB(clip)=-0.531 (thr 0.01)
v2_source_invisible_world_a              min_k_true_accept=None  min_k_false<=5%=None  any_accept_at_max_k=False  best_cal_LCB(clip)=-0.586 (thr 0.01)
```

## B2 k-curve (per world): accept rate, true/false/harmful, audit ΔbAcc, bounded cal-LCB (clipped), specificity
| world | k | n | acc_rate | true | false | harm | audit_ΔbAcc | cal_LCB_max(clip) | spec_cal | spec_aud |
|---|---|---|---|---|---|---|---|---|---|---|
| source_rich_source_vis | 1 | 2100 | 0.00 | 0 | 0 | 0 | n/a | n/a | 0 | 0 |
| source_rich_source_vis | 2 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| source_rich_source_vis | 4 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| source_rich_source_vis | 8 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| source_rich_source_vis | 16 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| source_rich_source_vis | 24 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| source_rich_source_vis | 32 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -0.884 | 0 | 0 |
| source_rich_source_vis | 40 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -0.695 | 0 | 0 |
| source_rich_source_vis | 50 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -0.531 | 0 | 0 |
| v2_source_invisible_wo | 1 | 2100 | 0.00 | 0 | 0 | 0 | n/a | n/a | 0 | 0 |
| v2_source_invisible_wo | 2 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| v2_source_invisible_wo | 4 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| v2_source_invisible_wo | 8 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| v2_source_invisible_wo | 16 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| v2_source_invisible_wo | 24 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -1.000 | 0 | 0 |
| v2_source_invisible_wo | 32 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -0.947 | 0 | 0 |
| v2_source_invisible_wo | 40 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -0.757 | 0 | 0 |
| v2_source_invisible_wo | 50 | 2100 | 0.00 | 0 | 0 | 0 | n/a | -0.586 | 0 | 0 |

## B3 sequential calibration (hardened bounded LCB)
- actions: {'abstain': 3180, 'reject': 1020}
- accepts: 0 ; false accepts: 0 ; k=1 accepts (must be 0): 0 ; mean label budget (accepted): None

## B4 oracle diagnostic (upper bound; excluded from deployable accept counts)
- source_rich_source_visible_wor: oracle audit ΔbAcc mean 0.021 / max 0.080 over 210 cells
- v2_source_invisible_world_a: oracle audit ΔbAcc mean 0.018 / max 0.080 over 210 cells

Budget curve: `tos_cmi/results/target_info/tier1_budget_frontier/tier1_budget_frontier_curve.png`

## Reading guide
- B0 source-only expected to abstain/reject on source-invisible benefit; B1 must NEVER accept (stop-cond).
- B2/B3 accept is the target-information signal: SAFE only if held-out audit ΔbAcc > +0.01 AND same-k random does not reproduce it (accepted_specific). accepted_non_specific / false_accept are disclosed.
- Many abstains at small k are EXPECTED (weak calibration LCB), not a failure.

