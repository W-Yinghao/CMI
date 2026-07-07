# Project B-Next Error-Risk Calibration Protocol

## 1. Purpose
Move Project B from a *support-valid identity* router toward a *risk-calibrated output* router by
building a unified source/target record layer and testing whether source-only **identity-error**
calibration is estimable where source-only **harm** calibration was not.

## 2. Why harm calibration is not enough
Source ACAR-harm needs source pseudo-domains where TTA is actually worse than identity. Those are
often single-class or absent, so harm calibration degenerates (`OACI_ACAR_HARM_CALIBRATION_DEGENERATE`)
and the router blocks TTA without a usable bound. That is a real non-identifiability, not a bug.

## 3. Why error calibration may help
Held-out source subjects/sites usually DO vary in identity error, so an error predictor has signal.
An error budget lets the router accept IDENTITY only when the calibrated upper error is acceptable,
directly targeting HF3's support-valid-but-concept-degraded identity.

## 4. Non-identifiability caveat
Error calibration inherits the SAME boundary as harm calibration. It may repair observable,
source-representative error modes (covariate / support / ESS), but it cannot solve concept-shift error
non-identifiability without a representativeness assumption linking source folds to the target: if the
concept relation differs on the target and leaves no covariate signature, a source-fold predictor sees
no error and accepts the degraded target. S0 TESTS this on HF3 rather than assuming it.

## 5. S0 record schema
Unified `source_or_target` rows share label-free diagnostics (density NLLs, support excess, ESS, ood,
prior_shift, min_class_responsibility, entropy/margin/max_prob, transform diagnostics) plus post-hoc
`identity_bacc`/`identity_error`. `label_access` is `source_calibration` (labels legal to fit a
predictor) or `target_posthoc` (labels for evaluation only).

## 6. Synthetic worlds
R2 (seeds 0,1,2), HF3 (3,4,7,8,10), H-OOD (32). fold_unit=site, record_unit=subject. HF3 is the probe:
if a source-site error predictor cannot flag concept-degraded target identity, ACAR-error hits the
same non-identifiability boundary.

## 7. Real BNCI2014_004 records
LOSO targets 1..4, subject+session eval, both support modes; source nested records from held-out source
subjects. Same schema as synthetic.

## 8. Toy error-risk probe
Numpy ridge on source records + split-conformal upper error (alpha=0.10); accept if
`upper_error <= error_budget` (0.45). Labelled explicitly a TOY predictor, not the final ACAR-error.

## 9. Label-safety rules
Target labels never fit a threshold, a predictor, or a decision; they enter only post-hoc metrics.
Source-calibration labels are legal for fitting.

## 10. What S0 can and cannot claim
S0 can claim: a correct, label-safe, source/target-separated record layer, and HF3 evidence for OR
against source-only error identifiability. S0 cannot claim a final ACAR-error router, accuracy gains,
or that concept shift is solved.
