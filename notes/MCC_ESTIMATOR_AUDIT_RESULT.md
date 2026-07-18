# MCC estimator audit — RESULT: K=4 is NOT variance-limited → default to risk-weighted MCC (NOT EMA)

Real EEG, 63/63 cells (27 BNCI2014 + 36 BNCI2015), full LOSO, at each bundle's hash-verified ERM warm-up
checkpoint. EXACT two-pass full-source MCC gradient (BN-frozen; verified == a single-graph full-batch reference to
rel<1e-5) vs K=4 / K=16 episodic estimators (R=64) + shuffled controls. NO model training. Manuscript FROZEN.

## Result (median over cells)
| dataset | n | A_4 (K4↔full align) | A_16 | A16−A4 gain | SNR_4→SNR_16 | dw_full/dw_k4 | frac(A_4<0.5) | true-vs-shuffle cos |
|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 | 27 | **+0.747** (mean 0.69, 63% >0.7) | +0.924 | +0.177 | 0.11→0.40 | **1.60** | 0.11 | +0.28 |
| BNCI2015_001 | 36 | **+0.811** (mean 0.81, 97% >0.7) | +0.945 | +0.134 | 0.23→0.71 | **1.73** | 0.00 | +0.56 |

## Reading — E1 (estimator variance-limited) is REJECTED
E1 required ALL of: median A_4 < 0.5 (both) AND A_16−A_4 > 0.25 (both) AND dw_full ≥ 2× dw_k4 (both) AND not
1–2-subject-driven. The data fail it decisively:
- **A_4 = 0.75 / 0.81 — far ABOVE the 0.5 variance-limited threshold** on both datasets (63% / 97% of cells >0.7;
  only 11% / 0% below 0.5). The K=4 episodic MCC gradient is a REASONABLY GOOD estimate of the exact full-source
  gradient — it is not noise-dominated.
- **dw_full / dw_k4 = 1.60 / 1.73 (< 2)** — even the EXACT full gradient moves the population WSCI only ~1.6–1.7×
  more than the noisy K=4 direction, nowhere near 2×; 0% / 8% of cells reach the 2× bar. So a zero-variance
  estimator would NOT unlock a materially larger geometry effect.
- K=16 does improve the estimate (A_16 ≈ 0.92–0.95, SNR ~2–3× K=4) — more samples help — but K=4 was already decent,
  and the improvement does not translate into a ≥2× population-geometry gain.

So the tiny, λ-inert, DG-null global-MCC effect is **NOT primarily an episodic-estimator-variance problem**. The
EMA / memory-bank prototype path is NOT earned.

## Disposition (per PM routing)
Default → **risk-weighted MCC**: stop treating all subject-class contrasts as equally important; weight the
consistency by SOURCE-ONLY predictive instability (leave-one-source-subject-out risk / class-margin reversal /
subject-specific rule residual), so the objective targets only the mechanism disagreement tied to source
generalization failure — consistent with the standing evidence that disagreement magnitude ≠ future harm (M1-P)
and that the global geometry axis is decoupled from DG (λ1 corr≈−0.05). EMA/prototype remains HELD (the audit did
not find the variance limitation that would motivate it).

## Honest caveats
- The audit is at the WARM-UP checkpoint (PM's chosen point — "where a misleading episodic estimator first bites");
  estimator quality along the 20-epoch continuation trajectory is not measured. This is by design (a frozen
  diagnostic) and does not affect the E1 rejection at the decision point.
- BNCI2014 shows some cell heterogeneity (11% of cells A_4<0.5, min 0.216) but the median/mean (0.69–0.75) and 63%
  >0.7 dominate; the E1 threshold is not met at the dataset level for either dataset.
- Per PM scope: a frozen diagnostic tests ONLY estimator quality, NOT geometry→DG. No training is committed by this
  result; the risk-weighted vs EMA fork is resolved toward risk-weighted, but the risk-weighting weight definition
  is to be frozen in a following contract before any GPU round.

HELD: EMA/prototype round, risk-weighted training (pending weight-definition contract), M2, learned projector, TTE,
CMI, manuscript. Scientific line ACTIVE.
