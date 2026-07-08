# Project B-Next ACAR-Error Protocol

## 1. Purpose
Upgrade the S0 toy error probe into a rigorous, records-level cross-fitted ACAR-error evaluation, to
decide whether source-only identity-error calibration is worth integrating into the router (S2B).

## 2. Difference from ACAR-harm
ACAR-harm needs source pseudo-domains where TTA is worse than identity (often single-class -> degenerate).
ACAR-error targets identity output eligibility: held-out source units vary in identity error, so an error
predictor has signal. Both share the same non-identifiability boundary for arbitrary target-only shift.

## 3. Why S0 records are enough for S2A
S0 froze source_nested_records (held-out source units, identity_error legal to fit) and target_eval_records
(post-hoc only). S2A re-uses them: no H2-CMI re-training, no new records.

## 4. Calibration modes
fold_local_crossfit  : deployment-faithful; per (config_id, support_mode, eval_unit) use only that fold's
                       source records; <3 source groups -> unavailable (no forced pooling).
pooled_world_crossfit: scientific-signal; per (dataset_or_world, support_mode, eval_unit) pool all source
                       records of that world/dataset. Not a single-target deployment guarantee.

## 5. Feature set and imputation
Core support/posterior features always used. TTA-transform features are optional and (per S0) all-NaN in
source calibration, hence dropped and audited. Imputation/scaling use SOURCE training statistics only;
target rows never inform fit/impute/scale/qhat/decision. All-NaN -> drop; zero-variance -> drop;
partially-missing -> impute from source mean; never silent 0.

## 6. Cross-fitting and conformal residuals
Group = (config_id, fold_unit_id). Leave-one-group-out: train ridge on source minus the held-out group,
predict it -> source OOF predictions. residual = max(0, true_error - oof_pred). Strict split-conformal
qhat at k=ceil((n+1)(1-alpha)); if k>n the strict bound is unavailable (state=unavailable_strict). A
relaxed quantile is reported for diagnostics only; the DECISION uses the strict bound.

## 7. Target decision simulation
upper_error = pred_error_target + qhat_strict. ACAR-error only ADDS a refusal layer:
acar_error_accept = support_accept AND upper_error <= error_budget (0.45). It never overrides a support
refusal. Post-hoc: violation = acar_error_accept AND true_error > budget.

## 8. HF3 boundary analysis
Central output: for concept-degraded HF3 identity (target_concept_hit AND identity_bacc<0.60), classify
caught_by_acar_error / boundary_confirmed_evaded_acar_error / support_already_refused / not_concept_degraded.

## 9. Real BNCI2014_004 analysis
Same schema. Fold-local is expected low-power (few source subjects -> unavailable); pooled-world gives a
weak signal that must not be overclaimed.

## 10. Claim boundary
S2A is a records-level evaluation, not a router. It can support (or refute) an ACAR-error output-eligibility
layer; it does not claim accuracy gains, does not solve concept shift, and does not remove the unified
non-identifiability boundary for arbitrary target-only harm or identity error.
