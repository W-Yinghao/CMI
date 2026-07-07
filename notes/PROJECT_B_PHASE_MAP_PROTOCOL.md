# Project B-Next Real EEG TTA Phase Map Protocol

## 1. Purpose
Map, on real EEG, where OFFLINE_TTA / PRIOR_ONLY help or harm vs IDENTITY, and decide the actionable
question: is the benefit phase SOURCE-PREDICTABLE.

## 2. Why source-predictability matters
A benefit phase is only router-actionable if predictable from source-only diagnostics / source-fold gain
records; otherwise the honest ceiling is refusal/identity governance (same non-identifiability wall as
harm and identity-error).

## 3. Datasets
Core: BNCI2014_004, BNCI2014_001. Optional availability-gated probe: Lee2019_MI. Bounded subjects/targets.

## 4. Actions compared
IDENTITY, OFFLINE_TTA (class-conditional affine), PRIOR_ONLY (S3A target-prior reweight). Evaluation only.

## 5. Diagnostics
Support (density NLL/threshold/excess), ESS, ood, prior_shift, entropy/margin/max_prob, TTA transform
(delta_density_nll/transform_norm/condition_number/pred_disagreement), ACAR error/harm states, v1 router.

## 6. Source-fold benefit prediction
Held-out source-subject folds give offline_tta_gain + diagnostics (labels legal = source calibration). A
deterministic numpy-ridge (reused error_risk cross-fit, target=offline_tta_gain) is fit source-only,
cross-fitted for source OOF, and applied to target for a transfer test.

## 7. Selective TTA policy simulation
select_tta if predicted gain > gain_margin (0.02); report source-OOF and target post-hoc selected gain /
harm / missed-benefit / caught-harm. Analysis thresholds are pre-declared, not tuned on results.

## 8. Label-safety
Target labels enter only post-hoc metrics; the benefit predictor is trained on source folds only;
imputation/scaling are source-only.

## 9. What S1A can claim
Whether a real-EEG benefit phase exists and whether it is source-predictable, per dataset/eval_unit/mode.

## 10. What S1A cannot claim
No accuracy SOTA, no router integration, no claim beyond the evaluated bounded datasets/subjects.
