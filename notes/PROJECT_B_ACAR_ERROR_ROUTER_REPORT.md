# Project B-Next ACAR-Error Router Report

*Optional ACAR-error output-eligibility integrated into RefusalFirstRouter. Records-only eval.*

## 1. Run status
- policies: support_only_v1, support_plus_acar_error_optional, support_plus_acar_error_required
- per-target decisions: 60 scope-summaries

## 2. Main result
Optional ACAR-error preserves support-valid R2 identity, catches most HF3 support-accepted concept-degraded identity, never enables TTA, and falls back to support-only when the error layer is unavailable (real BNCI2014_004).

## 3. Policy comparison (HF3 catch among support-accepted concept-degraded)

| calibration_mode | policy | n_cd | n_support_accepted | n_caught | catch_rate |
|---|---|---|---|---|---|
| fold_local_crossfit | support_plus_acar_error_optional | 16 | 7 | 7 | 1.000 |
| fold_local_crossfit | support_plus_acar_error_required | 16 | 7 | 7 | 1.000 |
| pooled_world_crossfit | support_plus_acar_error_optional | 16 | 7 | 5 | 0.714 |
| pooled_world_crossfit | support_plus_acar_error_required | 16 | 7 | 5 | 0.714 |

## 4. R2
- in_source_subject_q95/subject: err_layer=available cov=0.000 id_rate=0.000 add_refusal=0.000 viol=0.000 off_tta=0.000
- nested_site_excess_q95/subject: err_layer=available cov=0.833 id_rate=0.833 add_refusal=0.000 viol=0.000 off_tta=0.000
## 5. HF3
- in_source_subject_q95/subject: err_layer=available cov=0.000 id_rate=0.000 add_refusal=0.000 viol=0.000 off_tta=0.000
- nested_site_excess_q95/subject: err_layer=available cov=0.350 id_rate=0.350 add_refusal=0.250 viol=0.286 off_tta=0.000
## 6. H-OOD
- in_source_subject_q95/subject: err_layer=available cov=0.000 id_rate=0.000 add_refusal=0.000 viol=0.000 off_tta=0.000
- nested_site_excess_q95/subject: err_layer=available cov=0.000 id_rate=0.000 add_refusal=0.500 viol=0.000 off_tta=0.000
## 7. Real BNCI2014_004
- in_source_subject_q95/session: err_layer=unavailable cov=0.400 id_rate=0.400 add_refusal=0.000 viol=0.375 off_tta=0.000
- in_source_subject_q95/subject: err_layer=unavailable cov=0.500 id_rate=0.500 add_refusal=0.000 viol=0.000 off_tta=0.000
- nested_source_subject_excess_q95/session: err_layer=unavailable cov=0.400 id_rate=0.400 add_refusal=0.000 viol=0.375 off_tta=0.000
- nested_source_subject_excess_q95/subject: err_layer=unavailable cov=0.500 id_rate=0.500 add_refusal=0.000 viol=0.000 off_tta=0.000
- in_source_subject_q95/session: err_layer=unavailable cov=0.000 id_rate=0.000 add_refusal=0.400 viol=0.000 off_tta=0.000
- in_source_subject_q95/subject: err_layer=unavailable cov=0.000 id_rate=0.000 add_refusal=0.500 viol=0.000 off_tta=0.000
- nested_source_subject_excess_q95/session: err_layer=unavailable cov=0.000 id_rate=0.000 add_refusal=0.400 viol=0.000 off_tta=0.000
- nested_source_subject_excess_q95/subject: err_layer=unavailable cov=0.000 id_rate=0.000 add_refusal=0.500 viol=0.000 off_tta=0.000
- in_source_subject_q95/session: err_layer=unavailable cov=0.400 id_rate=0.400 add_refusal=0.000 viol=0.375 off_tta=0.000
- in_source_subject_q95/subject: err_layer=unavailable cov=0.500 id_rate=0.500 add_refusal=0.000 viol=0.000 off_tta=0.000
- nested_source_subject_excess_q95/session: err_layer=unavailable cov=0.400 id_rate=0.400 add_refusal=0.000 viol=0.375 off_tta=0.000
- nested_source_subject_excess_q95/subject: err_layer=unavailable cov=0.500 id_rate=0.500 add_refusal=0.000 viol=0.000 off_tta=0.000
- in_source_subject_q95/session: err_layer=unavailable cov=0.000 id_rate=0.000 add_refusal=0.400 viol=0.000 off_tta=0.000
- in_source_subject_q95/subject: err_layer=unavailable cov=0.000 id_rate=0.000 add_refusal=0.500 viol=0.000 off_tta=0.000
- nested_source_subject_excess_q95/session: err_layer=unavailable cov=0.000 id_rate=0.000 add_refusal=0.400 viol=0.000 off_tta=0.000
- nested_source_subject_excess_q95/subject: err_layer=unavailable cov=0.000 id_rate=0.000 add_refusal=0.500 viol=0.000 off_tta=0.000
## 8. Reason-code audit
Top codes: OACI_ACAR_INSUFFICIENT_CALIBRATION:590, OACI_ACAR_HARM_CALIBRATION_DEGENERATE:494, OACI_CONF_EMPTY_ACTION_SET:494, OACI_TOS_SUPPORT_MISMATCH:390, OACI_LEAKAGE_RESIDUAL_UNAVAILABLE:226, OACI_TTA_IDENTITY_FALLBACK:226, OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE:132, OACI_ACAR_HIGH_ACTION_RISK:122, OACI_PRIOR_SHIFT_ONLY_INFO:76
## 9. What this supports
An OPTIONAL ACAR-error eligibility layer that improves HF3 identity safety while preserving R2 and degrading gracefully on real low-power data.
## 10. What this does not support
It is not an accuracy claim, does not enable TTA, and does not remove the target-only non-identifiability boundary (H-OOD).
## 11. Recommendation
Adopt the optional policy as Project B-next deployment default; next best step is S3 PRIOR_ONLY (safest action to recover missed benefit), not S1 phase map.
