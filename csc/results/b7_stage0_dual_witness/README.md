# B7 Stage-0 dual-witness canary (diagnostic-only)

primary rule = old_B3_confirm AND B6_plain_confirm
exposed row-level re-analysis of committed B6.0 cohorts
development-only
not confirmatory
not deployable
not a universal type-I guarantee
B6-FM is secondary diagnostic only

## What this is
B6 as a single observational C-randomization null reached its structural limit (B6.0 covariate-correct/prior-wrong;
B6-FM within-class-covariate-wrong). B7-primary combines two COMPLEMENTARY witnesses via a nuisance-DISJUNCTION AND:
the OLD B3 fixed-margin Y-null = PRIOR-shift witness; the B6.0 plain C-null = COVARIATE-process witness. A concept
alert (DUAL_CONCEPT_ALERT) is allowed ONLY when BOTH nuisance explanations are rejected. Decision is STATE-based
(each witness's own gated confirm state); p_dual=max(p_old,p_C) is a DIAGNOSTIC only. 3-state:
DUAL_CONCEPT_ALERT / NO_ACTIONABLE_CONCEPT_EVIDENCE / UNIDENTIFIABLE_OR_INVALID (never NO_CONCEPT).

## Result (red-team CLEAN: rowjoin_state PASS + isolation_audit PASS, 0 serious)
B7-primary per condition: NULL_cov_soft 1, NULL_cov_plus_label 3, strong_0.81 0, strong_0.94 0, NULL_label 0,
random 0, POS_concept 7, POS_concept_plus_cov 7 -> development screen PASSED. The AND cancels each witness's blind
spot (NULL_label 0, strong-cov 0/0) and controls the hardest NCPL at 3 while retaining POS 7/7 (strong band).
DISJOINT partition (Option A, fail-closed) sums to 50/condition: both + old_only + B6_only + neither_valid + invalid.
(The earlier report mixed the four-cell with the invalid count; corrected to a single disjoint partition regenerated
from raw rows -- B7-primary counts unchanged.)

## Set-conservative (NOT a universal type-I guarantee)
The allow set is a SUBSET of old_B3 confirms AND of B6_plain confirms (verified b7==both<=min(old,plain)). Formal
validity depends on each component witness being valid for its component nuisance null. Honest residual: 4 dual
false-confirms on nulls (NULL_cov_plus_label 0020/0040/0046 x3; NULL_cov_soft #27 x1) within the dev screen; NCPL=3
is the flagged residual safety risk. NULL kinds reported separately (never pooled).

VERDICT: B7.0 passes the development screen on exposed data. Next (if authorized) = B7.1 exposed full dev replay
(8x300, same old_B3+B6_plain paths, same AND rule, no tuning; each null kind CP95 upper<0.05 no pooling). See
notes/b7_stage0_dual_witness.md. Related: b6_0_condition_randomization.md, b6_fm_condition_randomization.md.
