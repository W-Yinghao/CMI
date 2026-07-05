# C14 — EEG-DG Falsification Battery

> Support-aware leakage, selector-oracle replay, and source→target instability diagnostics as a reusable MEASUREMENT / FALSIFICATION instrument. OACI / SRC are NOT control methods.

- **CONTROL-HYPOTHESIS STATUS: `falsified`**
- falsification reasons: ['falsified_by_no_endpoint_transfer', 'falsified_by_oracle_failure', 'falsified_by_source_target_antitransfer']

## Gates

| gate | status |
|---|---|
| G0_integrity | `integrity_ok` |
| G1_selection_optimism | `selection_optimism_present` |
| G2_heldout_leakage | `weak_nominal_nonmultiplicity_signal` |
| G3_endpoint_transfer | `stop_no_reproducible_gain` |
| G4_oracle_rescue | `oracle_fails_to_rescue` |
| G5_source_target_transfer | `source_target_antitransfer_detected` |

## Evidence highlights

- **G0 integrity**: deep-verified=True, target_fit_empty=True, replay identity all-pass=True (argmax flips 0)
- **G1 selection optimism**: Δsel -0.3261 vs Δaudit +0.0076, transfer ratio -0.023, corr +0.004
- **G2 held-out leakage (K1)**: `weak_nominal_nonmultiplicity_signal` — 11 nominal, 0 BH survivors of 54
- **G3 endpoint (K2)**: `stop_no_reproducible_gain` (stop_no_reproducible_gain)
- **G4 oracle rescue**: `oracle_fails_to_rescue` (S5 oracle K2 = stop_no_reproducible_gain, source-only reproducing = none, S0=C8 check stop_no_reproducible_gain)
- **G5 source→target**: `source_target_antitransfer_detected` — ATI +1.000, instability score +1.000, anti-transfer 6/6 active cells, blowup 6

## Source→target instability

- source_nll→target_nll pearson -0.947 (near-zero/negative ⇒ source improvement does NOT reduce target loss)
- **anti-transfer index (ATI_NLL) +1.000**, severity (mean target-NLL harm) +0.8956, STI +1.000

## Method closure table

| hypothesis | status | next allowed action |
|---|---|---|
| OACI conditional-domain leakage-control | `closed_as_control_objective` | keep support-aware leakage + K1 as MEASUREMENT only; NO OACI-v2 selector, NO adversary tuning |
| SRC source-endpoint control | `closed_as_control_objective` | no further source-side endpoint control without a NEW transfer diagnostic |
| global_lpc / uniform (posterior-KL / uniform alignment) | `closed_as_positive_method` | retain as stress-test baselines only |
| support-aware extractable leakage (L_Q^ov) + K1 permutation null | `retained_as_measurement` | use inside the falsification battery; NOT as a control objective |

## Interpretation

> The experiments do not merely show that OACI underperforms ERM; they LOCALIZE the failure of the control hypothesis. Selection-time leakage reductions do not reliably survive audit; nominal audit leakage reductions do not produce endpoint gains; a source-audit oracle cannot rescue OACI trajectories; and a separate source-endpoint objective produces anti-transfer. Therefore the support-aware machinery should be retained as a FALSIFICATION and MEASUREMENT instrument, not as a control objective under this protocol.

**Say:** Under BNCI2014-001 LOSO with strict source/target isolation, the tested source-side control mechanisms do not transfer to target worst-domain endpoints. The measurement framework is useful precisely because it makes that failure visible.

**Do not over-claim:** ~~All DG is impossible.~~ / ~~EEG DG cannot work.~~ / ~~Support-aware invariance is useless.~~ / ~~OACI is mathematically wrong.~~

> **Verdict: the control hypothesis is FALSIFIED** (falsified_by_no_endpoint_transfer, falsified_by_oracle_failure, falsified_by_source_target_antitransfer). Retain support-aware leakage + K1/K2 + oracle replay + anti-transfer diagnostics as the falsification instrument; do NOT build another DG control penalty under this protocol.