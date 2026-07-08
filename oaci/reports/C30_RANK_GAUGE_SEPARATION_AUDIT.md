# C30 — Rank-Gauge Separation Audit (frozen C19 `664007686afb520f`)

> The competence signal has a within-target RANK axis (weakly source-visible) and a cross-target GAUGE axis (target-specific, source-unobservable, C23-C29). C30 separates them and attributes the rank to source families. Read-only over the C22 sidecar; no training/tuning/feature-selection. DIAGNOSTIC-ONLY.

- **PRIMARY: `G1_two_axis_rank_gauge_separation`** — the within-target RANK axis and the target-specific GAUGE axis are SEPARABLE (orthogonal; rank survives gauge control) -> the competence signal is two-axis.
- established: **G1_two_axis_rank_gauge_separation, G2_source_risk_carries_rank, G4_leakage_not_rank_carrier, G5_rank_signal_tracks_source_error_only, G7_unexplained_rank_residual**

## Q1 — rank vs gauge decomposition

- within-target RANK AUC **+0.659** vs pooled GAUGE AUC **+0.543** (target-centered recovers **+0.634**); rank⊥gauge orthogonality **-0.000**; gauge variance fraction **+0.536** → two-axis separation: **True**. within-target RANK axis AUC 0.659 (real) is orthogonal (corr -0.000) to the per-target GAUGE; pooled fails (0.543) but target-centering recovers (0.634) -> two-axis separation is real

## Q2 — which source family carries the within-target rank?

- probe-score rank strength |AUC−0.5| **+0.159**. Family best rank strength:

| family | best strength | mean strength | carries rank |
|---|---:|---:|:--:|
| source_risk | +0.124 | +0.109 | True |
| source_calibration | +0.089 | +0.054 | False |
| source_leakage | +0.039 | +0.031 | False |
| source_logit_geometry | +0.123 | +0.046 | True |

- top single family **source_risk** (gap over probe +0.034 is WITHIN bootstrap noise — NOT 'beats any family'); distributed in the RESIDUAL sense: **True**. **Transfer contrast**: multivariate probe sign-consistency **+1.000** (transfers: True) vs top-family sign-consistency **+0.778** (transfers: False). [RED-TEAM] the within-target rank's largest single carrier is the source_risk family (strength 0.124), but the score-vs-best-family gap (0.034) is WITHIN 9-target bootstrap noise -> NOT 'beats any family'. Distributedness holds in the RESIDUAL sense (score retains strength 0.113 after removing R_src). The MULTIVARIATE score is direction-CONSISTENT across targets (sign_consistency 1.00, transfers) while the top single family is target-LOCAL (sign-flips) -> the transferable within-target rank is DISTRIBUTED, not a single source family.

## Q4 — rank-gauge residualization

- within-target rank strength **+0.159** → after controlling R_src **+0.113**: survives **True** (gauge contaminates rank: **False**). the within-target rank SURVIVES controlling for R_src (rank strength 0.159 -> 0.113) -> the rank axis is separable, not a gauge/source-risk artifact

## Q3 — source-error alignment (RED-TEAM reworded: tautology + non-transfer)

- R_src↔source-NLL corr **+0.985** → 'tracks source error' is **TAUTOLOGICAL** (True); residualizing R_src on source NLL → strength **+0.016** (~chance, no target content beyond source risk).
- R_src within-target rank does NOT transfer: **2/9** targets on the majority side (sign-consistency **+0.778**, transfers **False**) → the 0.124 mean strength MASKS a target-LOCAL signal. [RED-TEAM] 'tracks source error' is a TAUTOLOGY: R_src IS the source NLL/CE risk (corr 0.985); residualizing R_src on the source NLL leaves strength 0.016 (~chance) -> R_src has no target-competence content beyond source risk. And the R_src within-target rank does NOT transfer: per-target AUC SIGN-FLIPS (7/9 targets on the majority side; sign_consistency 0.78), so the 0.124 mean strength MASKS a target-LOCAL, non-transferable signal. What survives: R_src is NOT a calibrated/deployable target-competence score.

## Q5 — C19 signal attribution

- within-target ranking supported: **True**; cross-target gauge supported: **False**; deployment selector established: **False**. C19's in-regime positive is the WITHIN-TARGET RANK axis (AUC 0.659); the cross-target GAUGE axis is NOT supported (pooled 0.543 fails; the gauge is source-unobservable per C23-C29). C19 is a weak within-target competence ranking signal, NOT a target-free detector and NOT a deployment selector.

## Red-team verification (5 independent adversarial checks on the real data)

- **G1 separation: CONFIRMED** — within-rank 0.659 sits ~15 SD above a within-target label-permutation null (p=0.005); epoch/order are 8–10× weaker and the rank SURVIVES controlling them (not a trajectory confound).
- **G2 R_src carries rank: CONFIRMED but WEAK** — beats 200/200 permutations, but same-family train_surrogate absorbs ~38%, and the R_src rank is TARGET-LOCAL (sign-flips across targets).
- **G4 leakage-not-carrier: NOT independently red-teamed this round** (asserted; leakage strength ~0.04).
- **G5 tracks-source-error: PARTIALLY REFUTED → reworded** — the 'tracks source error' leg is TAUTOLOGICAL (R_src ≡ source NLL); the rank does NOT transfer (per-target sign-flips). Only the negative conclusion (R_src is not a deployable competence score) survives.
- **G7 distributed: overfit attack REFUTED** (score is genuine out-of-target LOTO, p=0.00005) **but the score-minus-family gap is WITHIN 9-target bootstrap noise** — distributedness kept only in the RESIDUAL / sign-consistency sense (the multivariate probe transfers across targets; single families do not).

## Boundary of the claim

> DIAGNOSTIC-ONLY. Factor families FROZEN (no feature selection). The rank axis is a WEAK within-target competence ranking; the gauge axis needs target grouping (non-deployable) and is source-unobservable. rank+gauge is a diagnostic UPPER BOUND, NOT a selector. C19's positive is the RANK axis, NOT a target-free detector.