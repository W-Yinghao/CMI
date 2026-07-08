# Project B-Next Backend Comparison Report (S4A)

*Common-downstream fair comparison. h2cmi_common vs cbramod_common. NOT native h2cmi.*

## 1. Run status
- backends x dataset rows: 12

## 2. Availability
- BNCI2014_004/h2cmi_common: available=True 
- BNCI2014_004/cbramod_common: available=True 
- BNCI2014_001/h2cmi_common: available=True 
- BNCI2014_001/cbramod_common: available=True 
- Lee2019_MI/h2cmi_common: available=True 
- Lee2019_MI/cbramod_common: available=True 

## 3. Q1 identity bAcc (h2cmi_common vs cbramod_common)

| dataset | eval | id(h2cmi) | id(cbramod) | Δ |
|---|---|---|---|---|
| BNCI2014_001 | subject | 0.343316 | 0.303385 | -0.0399305 |
| BNCI2014_001 | session | 0.343316 | 0.303385 | -0.0399306 |
| BNCI2014_004 | subject | 0.656767 | 0.560526 | -0.096242 |
| BNCI2014_004 | session | 0.65628 | 0.561146 | -0.0951339 |
| Lee2019_MI | subject | 0.56625 | 0.4825 | -0.08375 |
| Lee2019_MI | session | 0.56625 | 0.4825 | -0.08375 |

## 4. Q2 support diagnostics (mismatch / excess)

| dataset | eval | backend | support_mismatch | excess | ess |
|---|---|---|---|---|---|
| BNCI2014_001 | session | cbramod_common | 0 | -35.6021 | 46.393 |
| BNCI2014_001 | subject | cbramod_common | 0 | -35.5921 | 92.8 |
| BNCI2014_001 | session | h2cmi_common | 0.375 | -0.514027 | 37.9405 |
| BNCI2014_001 | subject | h2cmi_common | 0.25 | -0.509397 | 75.881 |
| BNCI2014_004 | session | cbramod_common | 0.25 | -6.64765 | 66.0432 |
| BNCI2014_004 | subject | cbramod_common | 0.25 | -6.66869 | 332.129 |
| BNCI2014_004 | session | h2cmi_common | 0.75 | 2.19194 | 60.6911 |
| BNCI2014_004 | subject | h2cmi_common | 0.75 | 2.08115 | 310.332 |
| Lee2019_MI | session | cbramod_common | 0.125 | -39.5801 | 46.2214 |
| Lee2019_MI | subject | cbramod_common | 0 | -39.5785 | 92.7747 |
| Lee2019_MI | session | h2cmi_common | 0.875 | 4.01329 | 35.8729 |
| Lee2019_MI | subject | h2cmi_common | 0.75 | 4.01619 | 71.7458 |

## 5. Q3 ACAR-error transfer

| dataset | eval | backend | src_oof_err_corr | tgt_err_transfer | acar_accept | add_refusal |
|---|---|---|---|---|---|---|
| BNCI2014_001 | session | cbramod_common | 0.288814 | 0.815467 | 0 | 1 |
| BNCI2014_001 | subject | cbramod_common | 0.288814 | 0.939613 | 0 | 1 |
| BNCI2014_001 | session | h2cmi_common | 0.885857 | 0.841989 | 0 | 0.625 |
| BNCI2014_001 | subject | h2cmi_common | 0.885857 | 0.956842 | 0 | 0.75 |
| BNCI2014_004 | session | cbramod_common | 0.127353 | -0.0719399 | 0.1 | 0.65 |
| BNCI2014_004 | subject | cbramod_common | 0.127353 | -0.609371 | 0 | 0.75 |
| BNCI2014_004 | session | h2cmi_common | 0.665155 | 0.637705 | 0 | 0.25 |
| BNCI2014_004 | subject | h2cmi_common | 0.665155 | 0.915724 | 0 | 0.25 |
| Lee2019_MI | session | cbramod_common | 0.331173 | 0.315428 | 0 | 0.875 |
| Lee2019_MI | subject | cbramod_common | 0.331173 | 0.758933 | 0 | 1 |
| Lee2019_MI | session | h2cmi_common | -0.394711 | 0.0556914 | 0 | 0.125 |
| Lee2019_MI | subject | h2cmi_common | -0.394711 | 0.0546902 | 0 | 0.25 |

## 6. Q4 benefit phase / source-predictability / DEPLOYABILITY

A benefit phase is deployable only if post-TTA bAcc BEATS the best identity baseline (else a predictable gain is a weak-baseline artifact).

| dataset | eval | backend | tta_gain | exists | predictable | deployable | tta_bacc | best_id | -> |
|---|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 | session | cbramod_common | -0.0598957 | False | False | False | 0.24349 | 0.343316 | no_real_benefit_phase_observed |
| BNCI2014_001 | subject | cbramod_common | -0.0603299 | False | False | False | 0.243056 | 0.343316 | no_real_benefit_phase_observed |
| BNCI2014_001 | session | h2cmi_common | -0.0177951 | True | False | False | 0.325521 | 0.343316 | benefit_exists_but_not_source_predictable |
| BNCI2014_001 | subject | h2cmi_common | -0.0147569 | True | False | False | 0.328559 | 0.343316 | benefit_exists_but_not_source_predictable |
| BNCI2014_004 | session | cbramod_common | -0.059494 | True | False | False | 0.501652 | 0.65628 | benefit_exists_but_not_source_predictable |
| BNCI2014_004 | subject | cbramod_common | -0.059174 | False | False | False | 0.501351 | 0.656767 | no_real_benefit_phase_observed |
| BNCI2014_004 | session | h2cmi_common | -0.062872 | False | False | False | 0.593408 | 0.65628 | no_real_benefit_phase_observed |
| BNCI2014_004 | subject | h2cmi_common | -0.066214 | False | False | False | 0.590553 | 0.656767 | no_real_benefit_phase_observed |
| Lee2019_MI | session | cbramod_common | 0.015 | True | False | False | 0.4975 | 0.56625 | benefit_exists_but_not_source_predictable |
| Lee2019_MI | subject | cbramod_common | 0.01625 | True | True | False | 0.49875 | 0.56625 | predictable_but_weak_baseline_artifact |
| Lee2019_MI | session | h2cmi_common | -0.035 | False | False | False | 0.53125 | 0.56625 | no_real_benefit_phase_observed |
| Lee2019_MI | subject | h2cmi_common | -0.0225 | False | False | False | 0.54375 | 0.56625 | no_real_benefit_phase_observed |

## 7. Overall verdict
- **cbramod_weaker_representation_benefit_is_artifact**
- cbramod_common identity Δ vs h2cmi_common: -0.0731228 (negative = CBraMod weaker)
- cbramod source-predictable benefit: deployable=False, weak-baseline-artifact=True
## 8. Recommendation
CBraMod (zero-shot MI, common head) is a WEAKER representation (lower identity bAcc); its source-predictable TTA gain is a WEAK-BASELINE ARTIFACT -- CBraMod+TTA absolute bAcc stays BELOW the best identity baseline (h2cmi_common) on every dataset, so it is NOT deployable. S1A conclusion holds: keep h2cmi backend; refusal/identity governance; foundation zero-shot listed as bounded negative feasibility (fine-tuning is future work).
## 9. Boundary
Common Gaussian downstream is simpler than h2cmi's native head; absolute h2cmi_common numbers are a floor. CBraMod is applied zero-shot to MI. S4A isolates representation effects, not native systems or MI-tuned foundations.
