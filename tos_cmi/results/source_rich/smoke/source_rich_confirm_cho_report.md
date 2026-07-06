# Fork 2 Phase 1A --- source-rich smoke (Proposition 2 constructive test)

Thresholds FROZEN (safety<=0.020, benefit>0.010). E_oracle=regime (DIAGNOSTIC known-environment upper bound); E0 subject should MISS; E2/E4/E5 discovered (source-only); random = control. params {'frac': [0.4, 0.3, 0.3], 'noise': 0.1}.

| env | deployable? | ACCEPTs | target-good | HARMFUL accepts | benign-boundary | best benefit LCB (interv) | best target ΔbAcc [CI] |
|---|---|---|---|---|---|---|---|
| oracle | False | 6 | 5 | 0 | 1 | +0.293 (leace) | +0.056 [+0.037,+0.077] |
| subject | True | 0 | 0 | 0 | 0 | +0.001 (rlace) | +0.035 [+0.023,+0.049] |
| covariance_cluster | True | 6 | 5 | 0 | 1 | +0.073 (leace) | +0.056 [+0.037,+0.077] |
| margin_cluster | True | 0 | 0 | 0 | 0 | +0.000 (identity) | +0.000 [+0.000,+0.000] |
| augmentation_shift | True | 0 | 0 | 0 | 0 | +0.000 (identity) | +0.000 [+0.000,+0.000] |
| random | True | 0 | 0 | 0 | 0 | +0.000 (identity) | +0.000 [+0.000,+0.000] |

## Verdict
- E_oracle: target-good accepts 5 ; HARMFUL accepts 0 ; benign-boundary 1 (eps_coverage over-prediction, safe+target-positive)
- discovered (E2/E4/E5): target-good accepts 5 ; HARMFUL accepts 0 ; random target-good 0
- E0 subject baseline should MISS: subject target-good accepts = 0 (want 0 or << oracle)
- **Case A (E_oracle accepts AND a discovered source-only environment recovers it, beating random -> strongest source-only positive)**

Scatter (source-LOEO benefit LCB vs target ΔbAcc LCB, colored by environment): `tos_cmi/results/source_rich/smoke/source_rich_confirm_cho_scatter.png`

Proposition 2 empirically supported? YES -- E_oracle safely accepts >=1 target-beneficial intervention with 0 HARMFUL accepts (any benign-boundary accept is the eps_coverage slack at the strongest shift, not a violation)
