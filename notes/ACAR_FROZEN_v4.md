# ACAR_FROZEN_v4.md — v4 (CURB) candidate freeze **(DRAFT SKELETON — NOT YET TAGGED)**

```
STATE         : DRAFT SKELETON — binding ONLY when committed AND tagged `acar-v4-protocol` after explicit sign-off
LINEAGE       : v2 MEASUREMENT_ONLY (9b2f0c1) · v3 DEV_STOP (817b04f/9f4e83f) · v4 DEV candidate (e4c4e91, EXPLORATORY)
EXTERNAL ARM  : NOT RUN — authorized only AFTER this is tagged + a held-out cohort list is audited + signed off
LOCKBOX       : SEALED / NOT CONSUMED
DATE          : 2026-06-29
```

This freezes the SINGLE v4 candidate selected in `notes/ACAR_V4_CANDIDATE_SELECTION.md` from DEV exploration #001
(`results/acar_v4_dev_exploration_001/`). It is a **skeleton**: it fixes the recipe and the one confirmatory endpoint,
but it is **not** the binding protocol until this file is committed and tagged `acar-v4-protocol` with explicit sign-off,
and it authorizes **no** external read until a held-out cohort list is separately audited. v4 never edits the frozen
v2/v3 commits or tags.

## 1. Fixed substrate (inherited, unchanged)
Estimand `ΔR_a(B) = R_B(f_a) − R_B(f_0)` (NLL, paired, label-free at deployment; ΔR<0 = good). Actions `[identity,
matched_coral, spdim, t3a]`. Subject cluster `cohort_id::subject_id` is the calibration/eval unit. Batches B=32,
MIN_BATCH=8 (fallback retained, forced identity, in the denominator). Substrate = the v3 single-execution path
(`acar/v3` + `acar/v4/real_adapter.py`), reused unchanged.

## 2. The frozen candidate — ONE of everything
```
score_family   : shift_margin     harm = benefit = +features_v2[:, :, 1]  (= +d_margin; label-free, NEVER uses ΔR)
policy_family  : benefit_ranked   π(B) = identity if min_a benefit_a(B) > τ else argmin_a benefit_a(B);
                                   fallback batches forced identity. (direct_selective is numerically identical and is
                                   recorded as the non-primary alias.)
loss           : harm_indicator   L_s = mean_{B∈B(s)} 1[ π(B)≠identity ∧ ΔR_{π(B)}>0 ]   (fallback → 0, in denominator)
calibration    : finite-grid Learn-Then-Test (NO monotone-CRC theorem assumed)
                 method = ttest · correction = holm · alpha = 0.10 · budget(harm_indicator) = 0.10
                 grid_size = 12 · λ grid from CAL min-benefit statistics (acar.v4.develop._grid_for_family, label-free)
                 aggressiveness = increasing_lambda · select the MOST AGGRESSIVE passing λ
                 λ* PASS only if EVERY EVAL fold is certified (else NO_PASS)
weighting      : subject-macro (subject-equal); coverage/red/harm are weighted means; fallback in the denominator
splitting      : v3 S5 subject-disjoint outer folds → FIT/CAL; fold-local CAL→EVAL (λ* from CAL only); EXACT OOF coverage
provenance     : score_family_registry_sha256 = fe5a1f58986f7af1e8cb9db797ae9f08b46bfd749fc22ef8dbc8619005bc774e
                 DEV result manifest_sha256 = 8f5ccb288c7ca93857acd593ff6ec31bb4965c522a20d24b289ab9800bb970da
                 DEV v4_oof_records_sha256 = 7c7bcd51de874533cd75f9ec2ba64690930cd3bcbbb868bda7bcbce0a4909768
```
DEV #001 operating point (development; NOT a guarantee): PD cov 0.198 / red 0.116 / harm 0.154; SCZ cov 0.249 / red
0.201 / harm 0.205; macro red 0.158 > v2-replay 0.0985.

## 3. The SINGLE confirmatory endpoint (external Arm B — to be run only after tag + sign-off)
On each external held-out site, calibrate λ* on **site-local CAL subjects** with the frozen recipe (§2), deploy the
frozen policy on the held-out EVAL subjects, and evaluate the pre-registered, one-sided, subject-macro endpoint over
site-exchangeable subjects. **CONFIRMED** requires ALL of:
```
(a) harmful-adapted-batch rate controlled at the harm_indicator budget (0.10) on site-local CAL (the safety claim);
(b) deployed subject-macro NLL reduction > 0 AND > the site v2-replay comparator (utility beats the v2 baseline);
(c) adaptation coverage ≥ 0.15 (non-vacuous);
(d) both diseases non-vacuous if the site spans both; per-disease otherwise.
```
Result taxonomy (external): `V4_EXTERNAL_CONFIRMED` · `V4_EXTERNAL_NEGATIVE` · `OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT`.
No threshold/seed/loss/registry/grid change after the external read starts. A killed/partial run is operationally
aborted, never a verdict.

## 4. External held-out cohort list — TBD (NOT yet selected)
Required before Arm B: a metadata-only audit of held-out site(s) disjoint from the seven DEV cohorts (re-audit the v3
lockbox candidates; PD single-site contingency still applies). The chosen list, its disjointness proof, and the
site-local split rule are pinned HERE before any external read. **No external dataset is read until then.**

## 5. Discipline
This skeleton freezes nothing until (i) committed, (ii) tagged `acar-v4-protocol`, (iii) signed off, and (iv) §4 is
filled with an audited list. DEV exploration remains development-only; v4's authoritative status is EXPLORATORY_CANDIDATE
(Evidence Ledger A6) until an external `V4_EXTERNAL_CONFIRMED` exists. Never edit `9b2f0c1`, `817b04f`, `9f4e83f`, or any
v3 result.
