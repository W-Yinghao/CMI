# Project B-Next — Final Claim Boundary

Companion to `project_b_next_claim_boundary.json`.

## Claimable
1. Project B prevents unsafe adaptation under degenerate/unavailable ACAR-harm calibration.
2. Support-aware nested calibration can reduce synthetic over-refusal (R2 coverage 0.83, accepted bAcc 0.880).
3. Optional ACAR-error improves identity eligibility in source-representative error regimes (HF3 catch-among-support-accepted up to 1.00 fold-local).
4. Real-EEG bounded experiments show no deployable OFFLINE_TTA benefit phase for the current backend.
5. PRIOR_ONLY is lower harm than affine TTA (0.54 vs 0.71) but does not recover missed benefit.
6. Zero-shot CBraMod common backend does not create a deployable benefit phase.
7. OACI codes make refusal / identity / TTA-blocking decisions auditable.
8. Project B can serve as a hard safety governor inside an EEGAgent-style workflow.

## NOT claimable
1. It does not guarantee TTA improvement.
2. It does not solve arbitrary concept shift.
3. It does not identify target-only harm/error without a representativeness assumption.
4. It does not show full MOABB benchmark superiority.
5. It does not show foundation fine-tuning is ineffective (only zero-shot was tested).
6. It does not show EEGAgent itself improves decoding safety.
7. It does not claim target-label-tuned thresholds.
