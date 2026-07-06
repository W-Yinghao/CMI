# CSC B6.0 plain C-randomization canary — covariate-root fix with a prior-shift estimand failure

```
Scope: B6.0 plain condition-randomization null | development diagnostic only | NOT confirmatory | NO tag | NO validity claim
  root-cause redesign: randomize C ~ P(C|Z,S), NOT Y ~ h0(Y|Z) | replaces the CLOSED fitted-h0 line (B4)
  B3 contrast T + injection BYTE-UNCHANGED | propensity NEVER sees Y/synthetic/oracle | exact-CB Metropolis sampler
```

**Safe headline (red-team-verified):** *B6.0 inverts the null — it tests H0: Y ⊥ C | Z,S by resampling the CONDITION
C from an estimated covariate process P(C|Z,S) (exact-CB Metropolis odds-swap, count-preserving within subject) and
recomputing the byte-reused B3 statistic. It DEMONSTRABLY fixes the covariate-driven false-confirm that nothing else
could — strong-covariate nulls collapse from the old certifier's 18/50 (auc0.81) and 28/50 (auc0.94) to **0/50 and
0/50**, the soft covariate null from 11/50 to 2/50, destroyed-label control 0/50 — and it retains concept power. BUT
its estimand Y⊥C|Z is STRICTLY BROADER than concept shift: it also fires on pure PRIOR/label shift (NULL_label 0→25/50,
NULL_cov_plus_label 4→15/50), which the certifier's fixed-margin Y-null was specifically designed to EXCLUDE. This is a
verified estimand property, NOT a bug (destroyed-label control stays 0/50). B6.0 is a genuine PARTIAL advance; it is NOT
a validated concept certifier. Next = B6-FM (class-preserving / fixed-margin C-randomization).*

## Method (the inversion)
Old null (B3/B4): fix Z,C → draw Y\* ~ fitted h0(Y|Z) → recompute T. On real EEG the fitted-h0 null mis-centers /
under-disperses under covariate structure (P3 oracle; the strong-cov failure) — and estimating a deployable Y|Z null
was the un-closable problem (B4). B6 inverts it: **fix Z,Y → draw C\* ~ P(C|Z,S) → recompute the same B3 T(Y,Z,C\*)**.
P(C|Z,S) is estimable from ALL unlabeled trials — no label generator. Propensity v1 = cross-fit OOF L2 logistic on
subject-centered Z PCs (NO Y); resampler = exact conditional-Bernoulli Metropolis odds-swap chain (mirrors the
certifier's fixed-margin sampler), count-preserving within subject: flat propensity → within-subject permutation;
locked propensity → C\* freezes toward observed C → observed_T typical → abstain.

## Result (n=50/cond, base 200e6; OLD certifier vs B6 C-null; states re-derived from raw fields)
| condition | GT | OLD B3 | **B6 confirm** | B6 no-actionable | med p_C | med AUC | Cnull_Tz |
|---|---|---|---|---|---|---|---|
| NULL_cov_soft | covariate | 11 | **2** | 48 | 0.433 | 0.61 | +0.09 |
| NULL_cov_strong_auc0.81 | covariate | 18 | **0** | 50 | 0.764 | 0.98 | −0.73 |
| NULL_cov_strong_auc0.94 | covariate | 28 | **0** | 50 | 0.799 | 1.00 | −0.85 |
| random_label_control | (destroyed) | 0 | **0** | 50 | 0.520 | 0.61 | +0.05 |
| **NULL_label** | prior shift | 0 | **25** | 25 | 0.005 | 0.63 | **+3.21** |
| **NULL_cov_plus_label** | cov+prior | 4 | **15** | 35 | 0.052 | 0.62 | +1.77 |
| POS_concept | concept | 12 | 15 | 35 | 0.017 | 0.61 | +2.16 |
| POS_concept_plus_cov | concept | 13 | 12 | 38 | 0.010 | 0.61 | +3.33 |

## Reading (red-team-verified; accounting MINOR, decision_contract PASS, science MINOR/1-high)
1. **Covariate-root fix is REAL and p-value-driven** (NOT lock-masking; lock state fired 0/400). Under a strong
   covariate the propensity is near-deterministic (auc 0.98/1.00) → the CB odds-swap freezes C\* near observed C → the
   C-null PRESERVES the injected covariate structure → observed_T is a typical/below-bulk draw (Cnull_Tz −0.73/−0.85,
   obsT<Cnull mean in 70–72%). This is exactly the confound the OLD fixed-margin Y-null was blind to.
2. **Prior-shift firing is a CORRECT detection, NOT a bug.** NULL_label sets P(Y|session)=0.35/0.65 with P(Z|Y) fixed
   → by Bayes P(Y|Z,session) differs → Y genuinely NOT ⊥ C|Z. Decisive control: **random_label_control (labels
   destroyed) stays 0/50** (Cnull_Tz +0.05, p_C 0.52 — perfectly calibrated under the true null), so the firing is
   SPECIFIC to a real P(Y|C) difference. **Estimand duality:** the OLD Y-null is covariate-BLIND but prior-CORRECT;
   B6's C-null is covariate-CORRECT but prior-SENSITIVE.
3. **The safety failure is entirely the prior component.** Pure covariate soft = NULL_cov_soft 2/50 (passes);
   NULL_cov_plus_label 15/50 is its PRIOR part (not covariate leakage). This is a hard CSC-taxonomy failure
   (NULL_label / NULL_cov_plus_label must be controlled) — it BLOCKS any confirmatory concept-certifier claim.
4. **POS power retained but B6's count is admixed.** The CLEAN concept power is the prior-blind OLD null (12/50, 13/50);
   B6's 15/12 carries a small rotation-induced prior admixture — do NOT report B6's POS count as pure concept power.

## Accounting / disclosures (red-team-corrected)
- **n_total=400, n_valid_fidelity=399, n_invalid=1.** The 1 nan cohort (POS_concept #45) is an **OLD-certifier-side**
  degeneracy (its observed_T=nan) while B6's crt was VALID — a gate-fidelity gap (crt does not replicate the
  certifier's per-condition-class validity guard); blast-radius 1/400. (My earlier "crt needs ≥6 eligible" was wrong.)
- max|T_old−T_crt| = 0 over the 399; state-vs-p_C inconsistencies = 0; no B5 cache mixing; SM16 cache sha matches.
- **The UNIDENTIFIABLE_COVARIATE_LOCK state is INERT here** — eff_randomization is a global sum over ~5800 trials
  (min 171 ≫ floor 5), so it fired 0/400; the p-value carried the strong-null abstention. B6-FM must use a
  per-subject / count-conditioned lock measure.
- **No validity claim.** B6 validity depends on the estimated C|Z,S law; in-sample dim-reduction / cross-fit choices
  may affect calibration in a direction NOT assumed conservative.

## Verdict + next (reviewer directive)
**B6.0 plain C-null solves the strong-covariate false-confirm MECHANISM but FAILS the CSC null taxonomy (fires on
prior-shift controls).** Do NOT advance B6.1 plain-C-null; do NOT tag or claim as a validated certifier. **NEXT =
B6-FM: class-preserving / fixed-margin C-randomization** — restrict the Metropolis odds-swap to SAME-CLASS pairs within
subject (swap C between i=(C=hi,Y=y) and j=(C=lo,Y=y)) → preserves per-subject condition counts AND condition×class
margins → prior shift becomes a held-fixed nuisance while a rotated boundary still changes P(Z|Y,C) so POS still fires.
Y enters ONLY as an exact-margin constraint (never the propensity P(C|Z,S), never the score). A NEW design, NOT a
param-tune. Router/B5 PAUSED, B4 CLOSED, fitted-h0-repair / variance-inflation / p-recalibration / oracle-score /
confirmatory-tag all forbidden. Builds on [[csc-b6-condition-randomization]], [[p3_oracle_diagnostic]],
[[router_r1_scaleup]]. Package: `csc/results/b6_condition_randomization/b6_0_plain/`.
