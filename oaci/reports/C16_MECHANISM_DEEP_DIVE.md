# C16 — Mechanism & discriminative-validity deep dive (combined)

> Explains the measurement→control decoupling and source→target anti-transfer WITHOUT a new control objective. Real-data analyses read only committed C8/C10/C12 artifacts; the target oracle is NON-DEPLOYABLE.

## C16-A — Target-oracle ceiling
- **CASE `C3_calibration_not_discrimination`**: target oracle recovers accuracy but not calibration
- target-accuracy-good checkpoints exist (target oracle rescues bAcc: True) but are not source-observable (source oracle rescues: False); joint accuracy+calibration does not reproduce (stop_no_reproducible_gain).

## C16-B — Harm decomposition
- **`selected_oaci_calibration_improved_accuracy_flat`**: the SELECTED OACI is softer/better-calibrated (ΔNLL -0.0739, Δentropy +0.0803) but not more accurate (ΔbAcc -0.0024); class-boundary rotation True; subject-heterogeneous.
- SRC anti-transfer is MEMORIZATION: 6/6 cells flagged, mean index +1.965.

## Synthesis
> The measurement→control gap is not 'OACI is broken': the trajectory contains target-accuracy-good checkpoints that **source signal cannot observe**, and accuracy trades off against calibration so no single checkpoint jointly wins both. SRC's anti-transfer is source **memorization**. See C16_TARGET_ORACLE_CEILING.md, C16_HARM_DECOMPOSITION.md, C16_BATTERY_DISCRIMINATIVE_VALIDITY.md.

## C16-C — Battery discriminative validity
- positive controls certified: 1/1; negative controls falsified: 2/2; **discriminative_validity = True**