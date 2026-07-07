# Project B-Next PRIOR_ONLY Report

*PRIOR_ONLY action study (evaluation only). No router integration; no core change.*

## 1. Run status
- world-summary rows: 5; source-cal rows: 152

## 2. Main result
| world | eval | id | prior_only | offline_tta | po_gain | tta_gain | po_harm | tta_harm |
|---|---|---|---|---|---|---|---|---|
| BNCI2014_004 | session | 0.638 | 0.630 | 0.500 | -0.008 | -0.138 | 0.45 | 0.95 |
| BNCI2014_004 | subject | 0.637 | 0.632 | 0.500 | -0.005 | -0.137 | 0.75 | 1.00 |
| HF3 | subject | 0.558 | 0.537 | 0.498 | -0.020 | -0.060 | 0.65 | 0.50 |
| H_OOD | subject | 0.577 | 0.572 | 0.343 | -0.005 | -0.234 | 0.25 | 1.00 |
| R2 | subject | 0.815 | 0.770 | 0.958 | -0.045 | +0.143 | 0.58 | 0.08 |

## 3. R2
- subject: id=0.815 po=0.770 tta=0.958 | po_gain=-0.045 tta_gain=+0.143 | po_harm=0.58 tta_harm=0.08 | prior_only harmful; not safer than offline_tta
## 4. HF3
- subject: id=0.558 po=0.537 tta=0.498 | po_gain=-0.020 tta_gain=-0.060 | po_harm=0.65 tta_harm=0.50 | prior_only harmful; not safer than offline_tta
## 5. H-OOD
- subject: id=0.577 po=0.572 tta=0.343 | po_gain=-0.005 tta_gain=-0.234 | po_harm=0.25 tta_harm=1.00 | prior_only ~neutral; safer than offline_tta
## 6. Real BNCI2014_004
- session: id=0.638 po=0.630 tta=0.500 | po_gain=-0.008 tta_gain=-0.138 | po_harm=0.45 tta_harm=0.95 | prior_only ~neutral; safer than offline_tta
- subject: id=0.637 po=0.632 tta=0.500 | po_gain=-0.005 tta_gain=-0.137 | po_harm=0.75 tta_harm=1.00 | prior_only ~neutral; safer than offline_tta
## 7. Gain by prior shift

| world | eval | bin | n | po_gain | tta_gain | po_harm | tta_harm |
|---|---|---|---|---|---|---|---|
| BNCI2014_004 | session | low | 17 | -0.006 | -0.154 | 0.47 | 1.00 |
| BNCI2014_004 | session | medium | 3 | -0.021 | -0.048 | 0.33 | 0.67 |
| BNCI2014_004 | subject | low | 4 | -0.005 | -0.137 | 0.75 | 1.00 |
| HF3 | subject | high | 19 | -0.020 | -0.062 | 0.63 | 0.47 |
| HF3 | subject | medium | 1 | -0.015 | -0.008 | 1.00 | 1.00 |
| H_OOD | subject | high | 2 | +0.000 | -0.318 | 0.00 | 1.00 |
| H_OOD | subject | medium | 2 | -0.011 | -0.150 | 0.50 | 1.00 |
| R2 | subject | high | 4 | -0.080 | +0.304 | 1.00 | 0.00 |
| R2 | subject | low | 6 | -0.003 | +0.007 | 0.17 | 0.17 |
| R2 | subject | medium | 2 | -0.101 | +0.226 | 1.00 | 0.00 |

## 8. Source pseudo calibration
- source PRIOR_ONLY gain mean = -0.026, harm rate = 0.65 (n=152); source-only calibratability is reported, not assumed.
## 9. What this supports
PRIOR_ONLY is the lowest-harm adaptation action: harm rate=0.536667 vs OFFLINE_TTA=0.706667 (synthetic), real harm 0.6 vs 0.975; it is ~neutral/safe on H-OOD and real where OFFLINE_TTA is harmful.
## 10. What this does not support
It does NOT recover R2 missed benefit (R2 prior_only gain=-0.0451802, even in high-prior-shift bins) because R2's recoverable benefit is covariate-driven; not an accuracy claim; not a router integration; source PRIOR_ONLY harm is also high (not cleanly source-calibratable).
## 11. Recommendation
DEFER PRIOR_ONLY integration: PRIOR_ONLY IS lower-harm than OFFLINE_TTA, but GO condition(s) ['recovers/preserves_R2'] not met -> it does not recover/preserve R2 (the recoverable benefit there is covariate-driven, needing affine, not prior). Prefer S1 real phase map or backend comparison. [safer_than_offline_tta=True (harm 0.536667 vs 0.706667); recovers/preserves_R2=False (R2 gain -0.0451802); real_safer_than_affine=True (harm 0.6 vs 0.975)]
