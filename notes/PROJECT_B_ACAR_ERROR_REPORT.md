# Project B-Next ACAR-Error Report

*Cross-fitted, records-level ACAR-error evaluation. No router integration; no H2-CMI re-train.*

## 1. Run status
- scopes evaluated: 44
- calibration modes: fold_local_crossfit, pooled_world_crossfit

## 2. Main summary (pooled_world_crossfit)

| world | mode2 | supp | eval | n_src | n_tgt | state | qhat | oof_corr | supp_acc | acar_acc | supp_viol | acar_viol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BNCI2014_004 | pooled | in_source_subject_q95 | session | 8 | 20 | unavailable_strict | None | -0.020 | 0.400 |  | 0.375 |  |
| BNCI2014_004 | pooled | in_source_subject_q95 | subject | 8 | 4 | unavailable_strict | None | -0.020 | 0.500 |  | 0.000 |  |
| BNCI2014_004 | pooled | nested_source_subject_excess_q95 | session | 8 | 20 | unavailable_strict | None | -0.020 | 0.400 |  | 0.375 |  |
| BNCI2014_004 | pooled | nested_source_subject_excess_q95 | subject | 8 | 4 | unavailable_strict | None | -0.020 | 0.500 |  | 0.000 |  |
| HF3 | pooled | in_source_subject_q95 | subject | 80 | 20 | available | 0.091 | 0.595 | 0.000 | 0.000 | 0.000 | 0.000 |
| HF3 | pooled | nested_site_excess_q95 | subject | 80 | 20 | available | 0.091 | 0.595 | 0.600 | 0.350 | 0.583 | 0.286 |
| H_OOD | pooled | in_source_subject_q95 | subject | 16 | 4 | available | 0.154 | 0.875 | 0.000 | 0.000 | 0.000 | 0.000 |
| H_OOD | pooled | nested_site_excess_q95 | subject | 16 | 4 | available | 0.154 | 0.875 | 0.500 | 0.000 | 0.500 | 0.000 |
| R2 | pooled | in_source_subject_q95 | subject | 48 | 12 | available | 0.110 | 0.915 | 0.000 | 0.000 | 0.000 | 0.000 |
| R2 | pooled | nested_site_excess_q95 | subject | 48 | 12 | available | 0.110 | 0.915 | 0.833 | 0.833 | 0.000 | 0.000 |

## 3. R2
R2 = no concept shift; ACAR-error should preserve support-valid identity acceptance.
- in_source_subject_q95/subject: state=available oof_corr=0.915 supp_acc=0.000 acar_acc=0.000 acar_viol=0.000
- nested_site_excess_q95/subject: state=available oof_corr=0.915 supp_acc=0.833 acar_acc=0.833 acar_viol=0.000
## 4. HF3
HF3 = source-representative concept; central boundary test.

HF3 catch aggregate:

| mode | n_cd | supp_refused | caught | evaded | catch_rate_among_support_accepted |
|---|---|---|---|---|---|
| fold_local_crossfit | 16 | 9 | 7 | 0 | 1.000 |
| pooled_world_crossfit | 16 | 9 | 5 | 2 | 0.714 |

## 5. H-OOD
H-OOD = target-only concept (concept_frac=0.17); boundary expected to persist.
- in_source_subject_q95/subject: state=available oof_corr=0.875 supp_acc=0.000 acar_acc=0.000 acar_viol=0.000
- nested_site_excess_q95/subject: state=available oof_corr=0.875 supp_acc=0.500 acar_acc=0.000 acar_viol=0.000
## 6. Real BNCI2014_004
Expected low power; fold-local unavailable is acceptable.
- in_source_subject_q95/session: state=unavailable oof_corr= supp_acc=0.600 acar_acc= acar_viol=
- in_source_subject_q95/subject: state=unavailable oof_corr= supp_acc=1.000 acar_acc= acar_viol=
- nested_source_subject_excess_q95/session: state=unavailable oof_corr= supp_acc=0.600 acar_acc= acar_viol=
- nested_source_subject_excess_q95/subject: state=unavailable oof_corr= supp_acc=1.000 acar_acc= acar_viol=
- in_source_subject_q95/session: state=unavailable oof_corr= supp_acc=0.000 acar_acc= acar_viol=
- in_source_subject_q95/subject: state=unavailable oof_corr= supp_acc=0.000 acar_acc= acar_viol=
- nested_source_subject_excess_q95/session: state=unavailable oof_corr= supp_acc=0.000 acar_acc= acar_viol=
- nested_source_subject_excess_q95/subject: state=unavailable oof_corr= supp_acc=0.000 acar_acc= acar_viol=
- in_source_subject_q95/session: state=unavailable oof_corr= supp_acc=1.000 acar_acc= acar_viol=
- in_source_subject_q95/subject: state=unavailable oof_corr= supp_acc=1.000 acar_acc= acar_viol=
- nested_source_subject_excess_q95/session: state=unavailable oof_corr= supp_acc=1.000 acar_acc= acar_viol=
- nested_source_subject_excess_q95/subject: state=unavailable oof_corr= supp_acc=1.000 acar_acc= acar_viol=
- in_source_subject_q95/session: state=unavailable oof_corr= supp_acc=0.000 acar_acc= acar_viol=
- in_source_subject_q95/subject: state=unavailable oof_corr= supp_acc=0.000 acar_acc= acar_viol=
- nested_source_subject_excess_q95/session: state=unavailable oof_corr= supp_acc=0.000 acar_acc= acar_viol=
- nested_source_subject_excess_q95/subject: state=unavailable oof_corr= supp_acc=0.000 acar_acc= acar_viol=
- in_source_subject_q95/session: state=unavailable_strict oof_corr=-0.020 supp_acc=0.400 acar_acc= acar_viol=
- in_source_subject_q95/subject: state=unavailable_strict oof_corr=-0.020 supp_acc=0.500 acar_acc= acar_viol=
- nested_source_subject_excess_q95/session: state=unavailable_strict oof_corr=-0.020 supp_acc=0.400 acar_acc= acar_viol=
- nested_source_subject_excess_q95/subject: state=unavailable_strict oof_corr=-0.020 supp_acc=0.500 acar_acc= acar_viol=
## 7. Fold-local vs pooled-world interpretation
fold_local_crossfit is the deployment-faithful mode; pooled_world_crossfit is the scientific-signal mode and is NOT a single-target deployment guarantee.
## 8. Feature audit
- distinct feature statuses: ['dropped_all_nan', 'used']
- TTA-transform features (delta_density_nll/transform_norm/condition_number/pred_disagreement) status in source: ['dropped_all_nan']
## 9. What this supports
Cross-fitted ACAR-error carries source-representative identity-error signal that TRANSFERS to target: HF3 catch-rate-among-support-accepted=1.000 (tgt transfer corr high), and it preserves support-valid R2 identity (additional_refusal=0.000, R2 tgt corr=0.983).
## 10. What this does not support
It does not remove the target-only non-identifiability boundary: on H_OOD the predictor anti-transfers (target corr=-0.697), so any refusal there is an incidental conservative-margin effect, not identification. Not an accuracy claim; real BNCI2014_004 is low-power (strict conformal unavailable, n_source<9).
## 11. Next step recommendation
PROCEED to S2B: integrate ACAR-error as an optional output-eligibility layer in the RefusalFirstRouter (HF3 catch >=50%, R2 preserved, H_OOD boundary reported).
