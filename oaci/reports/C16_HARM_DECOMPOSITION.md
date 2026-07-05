# C16-B — Harm decomposition (OACI vs ERM target, logit-level from committed target_audit.npz)

- **selected-checkpoint verdict: `selected_oaci_calibration_improved_accuracy_flat`** over 54 fold-levels
- harm-type tally: {'discrimination_harm': 9, 'improved': 24, 'calibration_harm': 5, 'mixed_harm': 15, 'neutral': 1}
- aggregate Δ (OACI−ERM): bAcc -0.0024, NLL -0.0739, ECE -0.0355, entropy +0.0803, conf-on-wrong -0.0389, logit-norm -0.4171
- **class-boundary rotation: True** (some classes gain recall, others lose)

## Per-class recall Δ (OACI−ERM)

| class | Δ recall |
|---:|---:|
| 0 | +0.0266 |
| 1 | -0.0378 |
| 2 | +0.0334 |
| 3 | -0.0319 |

## Per-target-subject Δ (heterogeneous harm)

| target | Δ bAcc | Δ NLL |
|---:|---:|---:|
| 1 | -0.0095 | -0.1296 |
| 2 | -0.0035 | +0.1466 |
| 3 | -0.0353 | +0.0512 |
| 4 | +0.0333 | -0.4959 |
| 5 | -0.0081 | -0.0027 |
| 6 | +0.0041 | -0.1366 |
| 7 | -0.0124 | +0.0345 |
| 8 | -0.0165 | -0.0294 |
| 9 | +0.0263 | -0.1032 |

## SRC source-memorization index (from committed C12)

- **6/6 active cells flagged** as memorization (source NLL improves, target does not); mean memorization index **+1.965**

> The SELECTED OACI is softer/better-calibrated but not more accurate; combined with the C16-A target-oracle ceiling, the trajectory shows an ACCURACY<->CALIBRATION trade-off (accuracy-good checkpoints are calibration-worse). SRC anti-transfer is memorization (source NLL improves far more than target).