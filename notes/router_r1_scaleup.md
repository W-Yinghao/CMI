# CSC-realEEG Router R1 SCALE-UP — the router is a SOFT-covariate-regime tool with a HARD covariate-strength boundary

```
Scope: R1 scale-up (multi-block drift + strong-covariate null) | development diagnostic | NOT confirmatory | NO tag
  B3 certifier BYTE-UNCHANGED | router = post-hoc monotone threshold on observed_T | tau LOCKED (never re-derived)
  strong-cov = a SEPARATE injection module (does not mutate the frozen engine) | GT NO_CONCEPT by construction
```

**Safe headline (red-team-verified):** *The R1 abstention router controls held-out type-I ONLY in the soft-covariate
regime. (A) Under subject-subsampling of the same Lee2019 pool at the real (soft, session_auc≈0.52) covariate, the
locked-τ type-I is STABLE across 4 blocks (NULL_cov allow [5,1,5,3], all ≤7 cap; POS ~18%). (B) But a STRONG
covariate breaks it: a ground-truth-NO-concept null whose session covariate is amplified to auc 0.81 / 0.94 drives the
router's false-allow rate to 22/300 = 7.3% / 78/300 = 26% — well past the cap. An oracle-null test proves the observed
statistic is a typical draw from the TRUE null; the failure is the certifier's fixed-margin swap-null being
mis-centered ~7-10× and under-dispersed under a strong covariate. The router masks the fitted-null under-dispersion
only while the covariate is weak; it is NOT a general type-I controller.*

## (A) Multi-block drift — subsampling stability, NOT generalization
Fresh RNG-disjoint blocks 100e6/110e6/120e6 (+ original 90e6), locked τ=0.004587196, n=300/cond. Reproduced exactly:

| block | NULL_cov | NULL_cov+label | POS_concept |
|---|---|---|---|
| 90e6 (orig) | 5 | 4 | 52 |
| 100e6 | 1 | 6 | 61 |
| 110e6 | 5 | 3 | 56 |
| 120e6 | 3 | 4 | 54 |
| **max / CP95u** | **5 / 0.0347** | **6 / 0.0391** | ~18% |

All primary-null allows ≤ 7-cap. **Red-team correction (drift_validity MINOR_ISSUE):** the 4 "blocks" are NOT
independent datasets — every cohort resamples 30-of-54 subjects from ONE fixed cache at the SAME soft covariate
(session_auc 0.520–0.523, overlap 0.989 everywhere). So this is **Monte-Carlo subsampling stability on one dataset at
one weak stressor**, not distributional/covariate drift or cross-site generalization. Part A varies only the seed; the
real robustness question (covariate strength) is answered by (B). Do NOT read this as "the router generalizes."

## (B) Strong-covariate null — SOUND demonstration that type-I COLLAPSES (strongcov_validity PASS)
A ground-truth NO_CONCEPT null with an amplified session covariate: shift session-2's Z along the leading PC
**orthogonal to the pooled boundary w** (v−(v·ŵ)ŵ; verified w·v̂≈1e-14), draw Y\* from the boundary at the shifted Z.
Because w·Zp==w·Z, P(Y|Z) is session-invariant to machine precision → genuinely no concept; only P(Z|session) moves.

| delta | session_auc | method_confirm | **router_allow /300** | rate | T_z_med | ffp_floor | invalid |
|---|---|---|---|---|---|---|---|
| (0, anchor) | 0.53 | — | ≈5 (==NULL_cov) | 1.7% | — | — | 0 |
| **1.5** | **0.81** | 117 | **22** | **7.3%** | 10.6 | 1.00 | 0 |
| **2.5** | **0.94** | 173 | **78** | **26.0%** | 17.2 | 1.00 | 0 |

- **Genuine false-confirms, not abstentions:** valid=True, n_invalid=0 for all 600; confirms also defeat the
  subject-consistency gate (a systematic covariate → systematic per-subject interaction overfit). Monotone dose-response.
- **The STATISTIC is fine; the NULL is the problem (decisive oracle-null test):** regenerating Y\* from the TRUE
  boundary through the identical cross-fit puts observed T in the bulk of the true null (oracle_p≈Uniform), while the
  certifier's fixed-margin swap-null is mis-centered ~7-10× too low and 2-3× under-dispersed → fixed_margin_p pinned at
  the floor, T_z 10–28. A correctly-specified null does NOT reject; the fixed-margin parametric null does.
- **Mechanism:** the shift lands 100% in the certifier's rank-3 PC interaction basis (sessmean_in_top3PC=1.000); a
  strong marginal covariate inflates the finite-sample cross-fit interaction-overfit T (artifact-T up to ~0.0085) into
  the genuine-concept range (POS_concept median ~0.0097), so the observed_T≥τ gate that filtered the soft-covariate
  artifact (T<0.006) provably **cannot** filter it.

## Honest operating envelope
- **Robust to:** seed/subsample block-drift AND weak session covariates (auc≈0.52) — NULL false-confirm ≤5/300, POS ~18%.
- **NOT robust to:** covariate strength. Once the marginal shift occupies the certifier's leading-PC interaction basis
  (auc≥0.81), the under-dispersion is unmasked and type-I blows past the cap (7.3% → 26%).
- The τ gate is the entire type-I margin, calibrated on the same 54-subject pool (mild optimism: 80e6 calib→3, 90e6
  held-out→5); off-distribution transfer untested.

## Frozen-protocol eligibility — NOT as a general controller
The strong-cov failure is a HARD boundary, not a tunable margin. If ever frozen, the protocol MUST either (i) carry an
explicit **pre-deployment covariate-strength gate** (e.g. session_auc / marginal-shift bound below the auc0.81 breach),
or (ii) replace the fixed-margin swap-null with a **covariate-robust / true-law null** (the oracle test indicates that
restores control — but the true-generator oracle is not deployable, per [[p3_oracle_diagnostic]]). The cross-fit T
statistic is not indicted; the fixed-margin parametric null is.

## Disclosures (must accompany any use)
Single dataset (one Lee2019 54-subject cache); strong-cov collapse shown only for a covariate in the top-3 PC
interaction subspace (outside-subspace untested); one held-out block per strength; 3-point dose-response (thin,
direction unambiguous); mild calibration optimism.

## Relation
Ties the router's limit back to the SAME root cause as the whole line — the fitted-null under-dispersion
([[p3_oracle_diagnostic]], [[b4_stage1_canary]]). The router ([[router_stage1_validation]]) masks it in the
soft-covariate regime; the strong covariate un-masks it. [[b5_0_random_encoder]] / [[b5_1_ssl_encoder]] show richer
features (random or one learned SSL) don't fix the null either. Package: `csc/results/router_stage1_validation/scaleup/`.
