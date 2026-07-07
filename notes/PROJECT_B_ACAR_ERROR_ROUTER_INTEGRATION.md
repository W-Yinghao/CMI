# Project B-Next ACAR-Error Router Integration

## 1. Purpose
Integrate the S2A cross-fitted identity-error layer into the EXISTING RefusalFirstRouter as an OPTIONAL
output-eligibility gate, without changing the router policy core.

## 2. What S2A showed
Cross-fitted ACAR-error transfers on source-representative regimes (R2 preserved, HF3 caught) but
reproduces the non-identifiability boundary for target-only concept shift (H-OOD anti-transfer); real
BNCI2014_004 is low-power (strict conformal unavailable).

## 3. Integration policy
Three policies: support_only_v1; support_plus_acar_error_optional (require ACAR-error only when the layer
is AVAILABLE, else fall back to support-only); support_plus_acar_error_required (refuse when unavailable).

## 4. Optional vs required ACAR-error
Optional is the deployment default: it never turns an unavailable error layer into all-refuse. Required is
analysis-only and quantifies the coverage cost (expected to over-refuse real EEG).

## 5. How risk predictions enter RefusalFirstRouter
error_risk.make_identity_error_acar_state builds an ACARState (IDENTITY error only, no harm) from
cross-fitted OOF (true_error, pred_error) records; the target point prediction is passed via
risk_predictions; the router computes upper_error = pred + qhat and blocks IDENTITY when it exceeds the
error budget (OACI_ACAR_HIGH_ACTION_RISK). require_acar_error_for_output=True is set only when available.
Because OACI_ACAR_INSUFFICIENT_CALIBRATION is not an output blocker, the REQUIRED-on-unavailable refusal
is applied as an explicit policy wrapper in the eval script, not by mutating the router.

## 6. Label-safety
Error layer is fit on source records only (source-only imputation/scaling, cross-fitted). Target labels
enter only post-hoc metrics. TTA stays blocked under ACAR-harm degenerate/unavailable.

## 7. HF3 concept-degraded identity analysis
Per concept-degraded HF3 target: support_already_refused / caught_by_acar_error_router /
boundary_confirmed_evaded_acar_error_router.

## 8. H-OOD boundary
Target-only boundary persists (S2A anti-transfer); the router integration does not claim to fix it.

## 9. Real BNCI2014_004 low-power behavior
Fold-local error layer unavailable (few source subjects); optional == support-only; required over-refuses.

## 10. Claim boundary
S2B integrates an optional eligibility layer; it is not an accuracy claim and does not remove the
non-identifiability boundary. Deployment default is the OPTIONAL policy.
