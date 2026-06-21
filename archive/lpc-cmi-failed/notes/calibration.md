> ⛔ **DEMOTED (2026-06-21) — temperature/compression side-effect, NOT principled calibration.** Deconfound (`results/calibration_deconfound/`, 130 datasets): a single oracle temperature on ERM beats LPC NLL on **123/130** (LPC wins 7); LPC beats *raw* ERM (the table below, 115/130) but trivial rescaling does more, with acc ≈ ERM. The table's raw "ERM vs lpc" ΔECE/ΔNLL stands as recorded but does NOT support a "principled confidence regulariser" claim. The **TUAB** row is an **EXPOSED** dataset (`notes/TUAB_EXPOSURE_AUDIT.md`). See `notes/EVIDENCE_LEDGER.md` #9.

# Calibration (ECE% / NLL) — ERM vs lpc_prior, from saved `*.preds.npz` (no GPU/retrain)

_Lower is better. lpc_prior = smallest-λ (no-cost) config. Hypothesis: less subject-shortcut → better calibration._

| dataset | ERM ECE | lpc ECE | ΔECE | ERM NLL | lpc NLL | ΔNLL |
|---|---|---|---|---|---|---|
| ADFTD_Deep4Net | 42.0 | 37.6 | -4.4 | 3.499 | 2.394 | -1.105 |
| ADFTD_EEGNet_classbal | 32.3 | 22.8 | -9.4 | 2.183 | 1.514 | -0.669 |
| ADFTD_EEGNet_lamsweep | 32.7 | 32.3 | -0.4 | 2.186 | 1.982 | -0.204 |
| ADFTD_EEGNet_seed1 | 34.9 | 32.0 | -2.9 | 2.443 | 1.963 | -0.480 |
| ADFTD_EEGNet_seed2 | 29.0 | 27.4 | -1.6 | 1.996 | 1.864 | -0.132 |
| ADFTD_ShallowConvNet | 34.7 | 32.8 | -1.9 | 2.629 | 2.022 | -0.607 |
| ADFTD_bin_EEGNet_classbal | 12.0 | 10.1 | -1.9 | 0.667 | 0.635 | -0.032 |
| ADFTD_prior_subject_a1 | 32.8 | 23.4 | -9.5 | 2.197 | 1.508 | -0.689 |
| ADFTD_prior_trial_a0.1 | 32.8 | 22.3 | -10.4 | 2.196 | 1.484 | -0.712 |
| ADFTD_prior_trial_a1 | 32.0 | 23.1 | -8.9 | 2.160 | 1.500 | -0.660 |
| BNCI2014_001_EEGNet_lpcssl | 19.1 | 21.8 | +2.6 | 1.519 | 1.569 | +0.050 |
| BNCI2014_001_EEGNet_ssl | 19.1 | 21.9 | +2.8 | 1.517 | 1.572 | +0.054 |
| BNCI2014_004_EEGNet_imb_domainbal | 10.3 | 3.9 | -6.4 | 0.638 | 0.582 | -0.056 |
| BNCI2014_004_EEGNet_lpcssl | 7.8 | 6.1 | -1.8 | 0.626 | 0.635 | +0.009 |
| BNCI2014_004_EEGNet_ssl | 7.6 | 5.9 | -1.7 | 0.624 | 0.634 | +0.010 |
| BNCI2014_004_TSMNet_smalllam | 7.1 | 7.1 | -0.0 | 0.637 | 0.639 | +0.001 |
| DEAP_arousal_EEGNet | 28.0 | 18.4 | -9.6 | 1.095 | 0.835 | -0.260 |
| DEAP_quadrant_EEGNet | 34.5 | 16.4 | -18.0 | 2.315 | 1.513 | -0.802 |
| DEAP_quadrant_GraphCMI | 11.9 | 0.3 | -11.6 | 1.464 | 1.386 | -0.078 |
| MUMTAZ_EEGNet_sweep | 13.0 | 11.8 | -1.2 | 1.182 | 1.042 | -0.139 |
| TUAB_EEGNet | 29.3 | 24.8 | -4.5 | 1.679 | 1.200 | -0.479 |
| align_BNCI2014_001_LogCov_ha | 64.7 | 60.6 | -4.2 | 11.681 | 7.758 | -3.923 |
| align_BNCI2014_001_LogCov_none | 57.8 | 48.8 | -9.0 | 8.084 | 3.676 | -4.408 |
| align_BNCI2014_001_ea | 16.4 | 16.4 | -0.0 | 1.355 | 1.370 | +0.015 |
| align_BNCI2014_001_ea_strict | 28.3 | 28.6 | +0.3 | 1.955 | 1.915 | -0.041 |
| align_BNCI2014_001_none | 18.1 | 20.7 | +2.6 | 1.493 | 1.493 | +0.001 |
| align_BNCI2014_001_ra | 15.9 | 17.3 | +1.4 | 1.335 | 1.379 | +0.044 |
| align_BNCI2014_004_ea | 9.0 | 7.3 | -1.7 | 0.654 | 0.633 | -0.020 |
| align_BNCI2014_004_none | 7.2 | 7.6 | +0.4 | 0.621 | 0.632 | +0.011 |
| align_BNCI2014_004_ra | 9.3 | 7.1 | -2.2 | 0.653 | 0.634 | -0.019 |
| route2_BNCI2014_001_EEGNet | 18.5 | 18.3 | -0.2 | 1.413 | 1.421 | +0.008 |
| route2_BNCI2014_001_TSMNet | 35.9 | 35.4 | -0.5 | 2.365 | 2.267 | -0.098 |
| route2_BNCI2014_004_EEGNet | 7.8 | 3.1 | -4.7 | 0.613 | 0.580 | -0.033 |

*lpc_prior calibration ≤ ERM (ΔECE≤0.2) on 27/33 datasets.*