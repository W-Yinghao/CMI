# Project B-Next Backend Comparison Protocol (S4A)

## 1. Scientific question
Does a foundation-style EEG representation change the Project B decision problem (identity, support,
ACAR-error transfer, TTA benefit phase)?

## 2. Fair-comparison design
S4A is NOT native h2cmi system vs CBraMod system. It compares REPRESENTATIONS under a COMMON source-only
downstream. Both backends are frozen/source-trained encoders feeding an identical script-local pipeline:
source z-score + PCA (d<=min(128, raw_dim, n_source-K-1), source-fit only) + z-score;
class-conditional diagonal Gaussian generative classifier; PRIOR_ONLY reweight; common diagonal-affine
TTA; the same support/ESS/prior-shift/entropy diagnostics; and the S1A source-fold predictability test.

## 3. Absolute-number caveat
h2cmi_common uses the common Gaussian head, NOT h2cmi's native head, so its absolute numbers may be LOWER
than S1A native h2cmi. This is intentional to isolate representation effects. The apples-to-apples
comparison is h2cmi_common vs cbramod_common.

## 4. Backends
h2cmi_common: h2cmi encoder trained on source, frozen, embeddings -> common downstream.
cbramod_common: pretrained CBraMod (frozen), 200 Hz 1-second patches -> [B,C,P,200] flattened -> common
downstream. CBraMod is a general EEG foundation model applied ZERO-SHOT to MI; it is not MI-specialised.

## 5. Common affine TTA
Diagonal affine z' = z*exp(log_s)+b optimised on unlabeled target marginal NLL under the source Gaussian
+ L2(b)+L2(log_s); fixed hypers (steps=100, lr=0.01, lambda_b=0.01,
lambda_s=0.01); prior anchored post-hoc via pi_T shrinkage (tau=10.0); unstable/nonfinite ->
identity fallback with a recorded flag. No target-label tuning.

## 6. Predictability
Source-fold offline_tta_gain / identity_error predictors (reused error_risk cross-fit) give source OOF +
target transfer; a benefit phase is router-actionable only if source-predictable (transfer corr, selected
gain>0.02, selected harm<=0.25). CBraMod source folds refit only the common downstream (encoder frozen);
h2cmi source folds retrain the encoder (bounded max_nested_folds=2).

## 7. Label-safety
Target labels enter only post-hoc metrics; PCA/scaling/Gaussian/threshold/TTA use source only.

## 8. Availability
If CBraMod is unavailable, S4A records a feasibility failure (not a scientific negative) and reports
h2cmi_common only.

## 9. What S4A can / cannot claim
Can: whether the foundation representation changes identity/support/ACAR-error/benefit-phase under a
common head. Cannot: SOTA accuracy, a native-system comparison, or MI-specialised foundation performance.
