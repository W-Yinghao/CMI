# CSC B8.0 information-contract / randomized paired audit — design (development-only)

```
Scope: B8.0 information-contract | protocol + contract-validator + exact-randomization null + Lee2019 emulator
  development-only | NOT confirmatory | NOT deployable | NOT validation (Lee2019 is not a real randomized audit) | NO tag
  reviewer-authorized after B7.1 SAFETY-FAIL closed the observational null-repair line (B3->B7)
```

**Core inversion.** The observational line failed because C (condition) is confounded with Z on the mixed
covariate+prior cell, and any *estimated* null (Y|Z or C|Z,S) cannot separate that confound from concept. B8 stops
repairing the null: it **requires a known randomization contract** (C assigned by balanced randomization within
subject×block → C ⟂ Z within block by design) and uses an **exact block-stratified randomization null** (C\* from the
declared within-block randomization, byte-reused B3 contrast T(Y,Z,C\*)); when the contract is not met it **REFUSES**
(`CONTRACT_INVALID_OR_UNIDENTIFIABLE`) rather than false-alerting.

**Estimand (narrow).** Condition-specific boundary/conditional-structure evidence BEYOND label-prior shift, under a
randomized/counterbalanced paired audit contract — a condition×Z boundary interaction not explainable by randomized
assignment variation, label-prior margins, or block/time/session nuisance. Prior shift stays a nuisance.

**Contract (data-driven validator, predeclared, NOT from the test result — `check_contract`).**
(1) block-adjusted cross-fit AUC(C~Z) ≤ `TAU_CONTRACT_AUC=0.60` (C unpredictable from Z given block → randomized);
(2) ≥ `MIN_SUPPORT_BLOCKS=8` blocks with both conditions (randomization support). Fail → `CONTRACT_INVALID_OR_UNIDENTIFIABLE`.

**Exact null (`resample_C_exact`).** Within each (subject, block, class) stratum, uniform permutation of C preserving
the per-(block,class) counts EXACTLY — the KNOWN within-block randomization (class margins held → prior is nuisance).
NOT a fitted propensity. Recompute the same cross-fit T; p=(1+#{T\*≥T_obs}+inv)/(1+B); ALERT iff both p-gates ≤0.025
+ size.

**States:** `B8_CONCEPT_ALERT` / `NO_ACTIONABLE_CONCEPT_EVIDENCE` / `CONTRACT_INVALID_OR_UNIDENTIFIABLE` /
`INSUFFICIENT_LABELS` / `SAMPLER_INVALID`. Never `NO_CONCEPT`.

**Emulator (`build_b8_cohort`, Lee2019 SM16 geometry).** CONTRACT-SATISFYING (C balanced-randomized within block):
`CONTRACT_NULL_balanced`, `CONTRACT_NULL_prior_only`, `CONTRACT_NULL_cov_balanced`, `CONTRACT_random_label`,
`CONTRACT_POS_boundary`, `CONTRACT_POS_boundary_plus_prior` → expect nulls controlled + POS alerts. CONTRACT-VIOLATING
(C confounded with a label-free Z axis): `VIOLATION_cov_session`, `VIOLATION_prior_shift`, `VIOLATION_cov_plus_prior`
(the B7.1 killer), `VIOLATION_condition_lock` → expect `CONTRACT_INVALID` (correct REFUSAL), NOT false alerts.

**Canary (after design red-team clean):** n=50/cond, SM16, byte-reused B3 T, exact-randomization null, no feature
change, no learned encoder. Screen: contract-valid nulls ALERT ≤3/50; violation conditions `CONTRACT_INVALID` high AND
ALERT ≤3/50; `CONTRACT_POS_boundary` ALERT >0 (≥5 pref). If B8 can only reject via `CONTRACT_INVALID` with POS=0 →
contract too strong / label budget too small.

**Red-team focus (contract provenance):** blocks + assignment generated BEFORE labels/injection; assignment recorded +
seed/table hash-pinned; NO post-hoc assignment selection; C\* only from the predeclared within-block randomization;
block(+class) counts preserved; NO fitted Y|Z or C|Z propensity in the null; no oracle/router/observed_T threshold /
no B7 witness states; contract-invalid from predeclared checks; contract-valid vs violating separated, disjoint states,
no pooled claim.

Dev code: `realeeg_b8.py` (emulator + `check_contract` + `resample_C_exact` + `b8_certify`). Package (after canary):
`csc/results/b8_stage0_contract/`. Builds on [[csc-b8-information-contract]], [[b7_stage1_full_replay]].

---

## RESULT (post-run, 2026-07-06) — MIXED. Direction validated on emulator; NOT clean, NOT failure. NO tag.

12 worlds × 50 = 600 cohorts, base 400e6, SM16. Red-team `wtnmjmp1d`: accounting/provenance/isolation **PASS** (clean,
0 defects, 0 seed-intersection with 126 prior jsonl, 0/600 alert-decision mismatches, no fitted Y|Z or C|Z in null, no
oracle/router/B7-witness); science **MINOR_ISSUE** + 1 HIGH asterisk.

**KEY — the mixed covariate+prior cell that broke the observational line is controlled, by SPLITTING it (not a better null):**
- observational B7.1 `NULL_cov_plus_label` = **24/300 = 8% SAFETY-FAIL**
- B8 randomized form `CONTRACT_NULL_cov_plus_prior` = **ALERT 1/50 = 2%** (covariate drawn ⟂ C → exact null captures it)
- B8 confounded form `VIOLATION_cov_plus_prior` (AUC~0.78) = **REFUSED 50/50** (`CONTRACT_INVALID`)

Formulation-level correction (require+exploit randomization), not null-repair; argument rests on the PATTERN (all
well-behaved nulls balanced 0 / cov_balanced 1 / cov_plus_prior 1 / random 1, all ≤1/50) + 50/50 refusal of the confounded
forms — not the single 1/50 cell (CP95 overlaps B7.1's 8%). All 5 VIOLATION worlds correctly REFUSE (cov/cov+prior/no-rand
50/50 INVALID; borderline 4, lowvar 33 slip τ=0.60 but still ALERT **0/50** — exact null + rank-3 T robust to mild C~Z).

**Disclosed limitations (do NOT smooth over):**
1. **HIGH** — POS power does NOT yet exceed the collider-inflated null floor: POS_boundary 4/50=8% vs prior_only 5/50=10%,
   CP95 CIs fully overlap ([2.2,19.2]% vs [3.3,21.8]%). Cannot yet separate a weak concept from the prior-collider artifact
   at this effect size + n_eligible=30. "Safe-but-weak" only with this asterisk.
2. `CONTRACT_NULL_prior_only` 5/50 = approximate-exactness (Y is a post-treatment collider under `fix_class_margins`), the
   design-predicted limitation — a B8.1 target, not a p-value to retune.
3. Stress-design gap: no tested world = quiet-confound-passing-τ + C→Y prior effect; slip-through worlds have Y=draw(p0)
   (no concept), so "ALERT 0 on slip-through" ≠ proven robustness in the dangerous cell. UNTESTED → B8.1 requirement.
4. Emulator (Lee2019 SM16 geometry), semi-synthetic, n=50/world, single base seed — NOT real randomized-audit data.

**NEXT (reviewer decision, NOT yet authorized):** B8.1 = move class/prior balance INTO the randomization contract
(randomize C within subject×block×class → lower the collider floor) + an audit/label power budget (raise POS above the
floor). NOT another witness/variant, NOT threshold tuning, NOT a new statistic/encoder. Package: `csc/results/b8_stage0_contract/`.
