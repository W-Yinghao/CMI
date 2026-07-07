# Project B-Next PRIOR_ONLY Protocol

## 1. Purpose
Evaluate a minimal-intervention adaptation action, PRIOR_ONLY, as a candidate middle ground between the
too-conservative IDENTITY and the too-aggressive OFFLINE_TTA. Evaluation only; no router integration.

## 2. Why PRIOR_ONLY
OFFLINE_TTA is consistently harmful on real BNCI2014_004; IDENTITY leaves R2 missed benefit. PRIOR_ONLY
freezes encoder/density/classifier and only re-estimates the target class prior, matching Project B's
prior-decoupled design: prior shift alone should not refuse, but may justify a prior-only correction.

## 3. Action definition
Primary = identity posterior reweighting: p_prior(y|x) proportional to p_id(y|x) * pi_T(y)/pi_S(y),
renormalized. A diagnostic density-prior posterior (exp(logp)*pi_T) is recorded but is NOT primary.

## 4. Target prior estimation
pi_hat(y) = mean_i p_id(y|x_i) (unlabeled responsibilities); pi_T = (n*pi_hat + tau*pi_S)/(n+tau) with a
FIXED shrinkage tau=10.0 (not tuned on target). Shrinkage prevents small-batch prior collapse.

## 5. Posterior reweighting
Reweight the existing identity posterior by pi_T/pi_S; this is the least-intervention adaptation (no
encoder/density/affine update).

## 6. Label-safety
Target labels are never used to estimate pi_T, choose tau, or decide whether PRIOR_ONLY is applied; they
enter only the post-hoc bAcc/gain/harm block.

## 7. Synthetic worlds
R2 (0,1,2), HF3 (3,4,7,8,10), H-OOD (32); classes=3, sites=6, subjects=4, sessions=2, trials=60,
epochs=30, eval_unit=subject.

## 8. Real BNCI2014_004
LOSO targets 1..4, subject+session eval, epochs=8, CPU.

## 9. Source pseudo calibration
Nested source-site (synthetic) / source-subject (real, <=2 folds) PRIOR_ONLY gains, to test whether
PRIOR_ONLY harm is calibratable source-only.

## 10. What S3A can and cannot claim
S3A can claim whether PRIOR_ONLY is safer than OFFLINE_TTA, recovers R2 benefit, and is interpretable by
prior_shift. It cannot claim router integration, accuracy SOTA, or that PRIOR_ONLY is always beneficial.
