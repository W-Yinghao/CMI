# Project B-Next Real EEG TTA Phase Map Report

*Evaluation-only phase map + source-predictability test. No router integration.*

## 1. Run status
- datasets with rows: ['BNCI2014_001', 'BNCI2014_004', 'Lee2019_MI']

## 2. Dataset availability
- BNCI2014_004: available=True 
- BNCI2014_001: available=True 
- Lee2019_MI: available=True 

## 3. Main result

| dataset | eval | mode | id | tta | prior_only | tta_gain | tta_benefit% | tta_harm% |
|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 | session | in_source_subject_q95 | 0.351 | 0.250 | 0.326 | -0.101 | 0.00 | 0.88 |
| BNCI2014_001 | session | nested_source_subject_excess_q95 | 0.351 | 0.250 | 0.326 | -0.101 | 0.00 | 0.88 |
| BNCI2014_001 | subject | in_source_subject_q95 | 0.351 | 0.250 | 0.326 | -0.101 | 0.00 | 1.00 |
| BNCI2014_001 | subject | nested_source_subject_excess_q95 | 0.351 | 0.250 | 0.326 | -0.101 | 0.00 | 1.00 |
| BNCI2014_004 | session | in_source_subject_q95 | 0.639 | 0.500 | 0.630 | -0.139 | 0.00 | 0.90 |
| BNCI2014_004 | session | nested_source_subject_excess_q95 | 0.639 | 0.500 | 0.630 | -0.139 | 0.00 | 0.90 |
| BNCI2014_004 | subject | in_source_subject_q95 | 0.638 | 0.500 | 0.632 | -0.138 | 0.00 | 1.00 |
| BNCI2014_004 | subject | nested_source_subject_excess_q95 | 0.638 | 0.500 | 0.632 | -0.138 | 0.00 | 1.00 |
| Lee2019_MI | session | in_source_subject_q95 | 0.548 | 0.500 | 0.535 | -0.048 | 0.00 | 0.62 |
| Lee2019_MI | session | nested_source_subject_excess_q95 | 0.548 | 0.500 | 0.535 | -0.048 | 0.00 | 0.62 |
| Lee2019_MI | subject | in_source_subject_q95 | 0.548 | 0.500 | 0.539 | -0.048 | 0.00 | 0.50 |
| Lee2019_MI | subject | nested_source_subject_excess_q95 | 0.548 | 0.500 | 0.539 | -0.048 | 0.00 | 0.50 |

## 4. BNCI2014_004
- session/in_source_subject_q95: tta_gain=-0.139167 benefit_rate=0 harm_rate=0.9 | exists=False predictable=False transfer_corr=0.608564 -> no_real_benefit_phase_observed
- session/nested_source_subject_excess_q95: tta_gain=-0.139167 benefit_rate=0 harm_rate=0.9 | exists=False predictable=False transfer_corr=0.610783 -> no_real_benefit_phase_observed
- subject/in_source_subject_q95: tta_gain=-0.137811 benefit_rate=0 harm_rate=1 | exists=False predictable=False transfer_corr=0.899613 -> no_real_benefit_phase_observed
- subject/nested_source_subject_excess_q95: tta_gain=-0.137811 benefit_rate=0 harm_rate=1 | exists=False predictable=False transfer_corr=0.899443 -> no_real_benefit_phase_observed
## 5. BNCI2014_001
- session/in_source_subject_q95: tta_gain=-0.100694 benefit_rate=0 harm_rate=0.875 | exists=False predictable=False transfer_corr=nan -> no_real_benefit_phase_observed
- session/nested_source_subject_excess_q95: tta_gain=-0.100694 benefit_rate=0 harm_rate=0.875 | exists=False predictable=False transfer_corr=nan -> no_real_benefit_phase_observed
- subject/in_source_subject_q95: tta_gain=-0.100694 benefit_rate=0 harm_rate=1 | exists=False predictable=False transfer_corr=nan -> no_real_benefit_phase_observed
- subject/nested_source_subject_excess_q95: tta_gain=-0.100694 benefit_rate=0 harm_rate=1 | exists=False predictable=False transfer_corr=nan -> no_real_benefit_phase_observed
## 6. Lee2019_MI optional probe
- session/in_source_subject_q95: tta_gain=-0.0475 benefit_rate=0 harm_rate=0.625 | exists=False predictable=False transfer_corr=0.653089 -> no_real_benefit_phase_observed
- session/nested_source_subject_excess_q95: tta_gain=-0.0475 benefit_rate=0 harm_rate=0.625 | exists=False predictable=False transfer_corr=0.653089 -> no_real_benefit_phase_observed
- subject/in_source_subject_q95: tta_gain=-0.0475 benefit_rate=0 harm_rate=0.5 | exists=False predictable=False transfer_corr=0.689236 -> no_real_benefit_phase_observed
- subject/nested_source_subject_excess_q95: tta_gain=-0.0475 benefit_rate=0 harm_rate=0.5 | exists=False predictable=False transfer_corr=0.689236 -> no_real_benefit_phase_observed
## 7. Benefit phase analysis

| dataset | eval | mode | exists | predictable | transfer_corr | sel_gain | sel_harm | implication |
|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 | session | in_source_subject_q95 | False | False | nan | nan | 0 | no_real_benefit_phase_observed |
| BNCI2014_001 | session | nested_source_subject_excess_q95 | False | False | nan | nan | 0 | no_real_benefit_phase_observed |
| BNCI2014_001 | subject | in_source_subject_q95 | False | False | nan | nan | 0 | no_real_benefit_phase_observed |
| BNCI2014_001 | subject | nested_source_subject_excess_q95 | False | False | nan | nan | 0 | no_real_benefit_phase_observed |
| BNCI2014_004 | session | in_source_subject_q95 | False | False | 0.608564 | -0.139167 | 0.9 | no_real_benefit_phase_observed |
| BNCI2014_004 | session | nested_source_subject_excess_q95 | False | False | 0.610783 | -0.139167 | 0.9 | no_real_benefit_phase_observed |
| BNCI2014_004 | subject | in_source_subject_q95 | False | False | 0.899613 | -0.137811 | 1 | no_real_benefit_phase_observed |
| BNCI2014_004 | subject | nested_source_subject_excess_q95 | False | False | 0.899443 | -0.137811 | 1 | no_real_benefit_phase_observed |
| Lee2019_MI | session | in_source_subject_q95 | False | False | 0.653089 | -0.0475 | 0.5 | no_real_benefit_phase_observed |
| Lee2019_MI | session | nested_source_subject_excess_q95 | False | False | 0.653089 | -0.0475 | 0.5 | no_real_benefit_phase_observed |
| Lee2019_MI | subject | in_source_subject_q95 | False | False | 0.689236 | -0.0025 | 0 | no_real_benefit_phase_observed |
| Lee2019_MI | subject | nested_source_subject_excess_q95 | False | False | 0.689236 | -0.0025 | 0 | no_real_benefit_phase_observed |

## 8. Source-predictability / transfer
- BNCI2014_001/session/in_source_subject_q95: predictor=True oof_corr=nan target_transfer_corr=nan target_select_rate=0 target_selected_gain=nan
- BNCI2014_001/session/nested_source_subject_excess_q95: predictor=True oof_corr=nan target_transfer_corr=nan target_select_rate=0 target_selected_gain=nan
- BNCI2014_001/subject/in_source_subject_q95: predictor=True oof_corr=nan target_transfer_corr=nan target_select_rate=0 target_selected_gain=nan
- BNCI2014_001/subject/nested_source_subject_excess_q95: predictor=True oof_corr=nan target_transfer_corr=nan target_select_rate=0 target_selected_gain=nan
- BNCI2014_004/session/in_source_subject_q95: predictor=True oof_corr=0.292686 target_transfer_corr=0.608564 target_select_rate=1 target_selected_gain=-0.139167
- BNCI2014_004/session/nested_source_subject_excess_q95: predictor=True oof_corr=0.292686 target_transfer_corr=0.610783 target_select_rate=1 target_selected_gain=-0.139167
- BNCI2014_004/subject/in_source_subject_q95: predictor=True oof_corr=0.292686 target_transfer_corr=0.899613 target_select_rate=1 target_selected_gain=-0.137811
- BNCI2014_004/subject/nested_source_subject_excess_q95: predictor=True oof_corr=0.292686 target_transfer_corr=0.899443 target_select_rate=1 target_selected_gain=-0.137811
- Lee2019_MI/session/in_source_subject_q95: predictor=True oof_corr=0.285522 target_transfer_corr=0.653089 target_select_rate=1 target_selected_gain=-0.0475
- Lee2019_MI/session/nested_source_subject_excess_q95: predictor=True oof_corr=0.285522 target_transfer_corr=0.653089 target_select_rate=1 target_selected_gain=-0.0475
- Lee2019_MI/subject/in_source_subject_q95: predictor=True oof_corr=0.285522 target_transfer_corr=0.689236 target_select_rate=0.5 target_selected_gain=-0.0025
- Lee2019_MI/subject/nested_source_subject_excess_q95: predictor=True oof_corr=0.285522 target_transfer_corr=0.689236 target_select_rate=0.5 target_selected_gain=-0.0025

**Offset-transport failure:** where the target transfer correlation is high yet the source-selected target gain is <= 0, the predictor RANK-transfers but its OFFSET does not — a naive selective-TTA policy would select HARMFUL TTA. Affected: BNCI2014_004/session/in_source_subject_q95, BNCI2014_004/session/nested_source_subject_excess_q95, BNCI2014_004/subject/in_source_subject_q95, BNCI2014_004/subject/nested_source_subject_excess_q95, Lee2019_MI/session/in_source_subject_q95, Lee2019_MI/session/nested_source_subject_excess_q95, Lee2019_MI/subject/in_source_subject_q95, Lee2019_MI/subject/nested_source_subject_excess_q95. This is the same non-identifiability boundary (score offset not source-calibratable) seen for harm and identity error.
## 9. Router implications
- overall verdict: **no_real_benefit_phase_observed**
- No OFFLINE_TTA benefit phase exists on any evaluated real dataset (benefit rate 0; TTA harmful/neutral). Project B v1 selected OFFLINE_TTA on 0 domains, so its refusal/identity routing is CORRECT, not overconservative — there is no benefit it wrongly refuses, and a source-only selective policy would actively select harmful TTA (offset-transport failure).
## 10. Recommendation
Stop chasing TTA on this backend; next = foundation-model backend comparison or manuscript consolidation.
