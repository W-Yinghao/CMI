# C87_CONTROL_PASS — signed token

```text
TOKEN               : C87_CONTROL_PASS
SIGNATURE           : b31cd7314b1a4c59
GATE CODE COMMIT    : 54485437 (oaci/active_testing/c87/, branch oaci)
RESULT ARTIFACT     : oaci/reports/C87_CONTROL_GATE_RESULT.json
RUN                 : K_MC=1000; A=648; pos_n_pat=1000 (power/specificity), n_pat=400 (coverage stress);
                      B_boot=2000; alpha=0.05; tau_G=0.25; elapsed 4613.6s
CONTROLS (all PASS) : POS (power 0.806>=0.80; gain @MODEL-SELECTOR,B=16) ; POS_DENSE (winner's-curse
                      corrected) ; NEG_A (no false transport) ; NEG_B (real-gate FP rate 0.0000<=alpha;
                      null 100% vacuous) ; CALIB (gate-statistic coverage 0.960 in band) ;
                      MUTATIONS M1/M2/M3 (all trigger)
```

## What this token DOES and does NOT mean
- DOES: the C87 PRODUCTION pipeline (v3 held-selection cross-fit estimand, LURE, P0/LURE-AT/
  MODEL-SELECTOR/CODA policies, patient-cluster CRN-paired BCa bootstrap, IUT+Holm gate with the
  vacuity/Georgia verdict caps) provably DETECTS a planted positive (POS/POS_DENSE), REFUSES to
  manufacture one (NEG_A/NEG_B), and is CALIBRATED (CALIB) — validated on SYNTHETIC worlds only.
  It is a NECESSARY engineering precondition for C87E, per C87P S6.6.
- DOES NOT: it is NOT a scientific result about ECG; it does not predict the real outcome, tune any
  real threshold, or substitute for the untouched-cohort gate.

## Remaining pre-C87E steps (NOT closed by this token)
- S3.4 reference-CODE-agreement audit: match MODEL-SELECTOR/CODA against the public repos
  (okanovic/model-selector, justinkay/coda) — repos are not local; this token validates statistical
  BEHAVIOR, not byte-level reference agreement.
- The two frozen metadata pre-conditions (signed support-level-1 deletion-vs-task manifest; unique
  scored-class registry+hash resolving 26 vs 30).
- HONEST MARGIN NOTE: POS joint power = 0.806, just above the 0.80 floor (thin margin; a small-sample
  pre-check had estimated ~0.94 — the full K_MC=1000 estimate is the authoritative 0.806).

## Status
C87_CONTROL_PASS produced. C87E real-data download/training/outcome-access NOT authorized. C88 NOT authorized.
