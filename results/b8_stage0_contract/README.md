# B8.0 information-contract / randomized paired audit canary (diagnostic-only) — MIXED

Core inversion: stop repairing observational nulls. **Require a KNOWN randomization contract** (C assigned by balanced
randomization within subject×block → C ⟂ Z within block by design) + an **EXACT block-stratified randomization null**
(C\* from the declared within-block randomization; byte-reused B3 contrast T(Y,Z,C\*)); **REFUSE** (`CONTRACT_INVALID_OR_UNIDENTIFIABLE`)
when the contract is unmet. Reviewer-authorized after B7.1 SAFETY-FAIL closed the observational line (B3→B4→B5→B6→B7).

development-only / NOT confirmatory / NOT deployable / **NOT validation (Lee2019 SM16 geometry is a realistic substrate,
not a real randomized audit)** / no tag / no in-place retuning. 12 worlds × 50 = 600 cohorts, base 400e6, SM16 features.

## RESULT — MIXED (red-team: accounting/isolation **PASS** clean; science **MINOR_ISSUE** + 1 HIGH asterisk)

**KEY (the point of B8):** the mixed covariate+prior cell that structurally defeated the observational line is
**controlled once the randomization is known** — and B8 does it by *splitting* that cell, not by a better null:

| | observational (B7.1) | B8 randomized form | B8 confounded form |
|---|---|---|---|
| mixed covariate+prior | `NULL_cov_plus_label` **24/300 = 8% SAFETY-FAIL** | `CONTRACT_NULL_cov_plus_prior` **ALERT 1/50 = 2%** | `VIOLATION_cov_plus_prior` (AUC~0.78) **REFUSED 50/50** |

Under a valid within-block contract the covariate is drawn *independently* of C, so the exact null captures it; where
the data are actually confounded, B8 **refuses** rather than false-alerting. This is a formulation-level correction
(require+exploit randomization), **not** "a better null repairs the same broken data."

**Per-world (contract-valid vs contract-violating reported SEPARATELY — no pooling):**

```
CONTRACT-VALID NULLS (want ALERT <=3/50):
  CONTRACT_NULL_balanced          ALERT 0/50   (valid 50)
  CONTRACT_NULL_cov_balanced      ALERT 1/50   (valid 50)
  CONTRACT_NULL_cov_plus_prior    ALERT 1/50   (valid 50)   <- B7.1-killer, randomized form
  CONTRACT_random_label           ALERT 1/50   (valid 50)
  CONTRACT_NULL_prior_only        ALERT 5/50   (valid 50)   <- OVER-ALERT (collider, see below)
CONTRACT-VALID POS (want ALERT >0, >=5 strong):
  CONTRACT_POS_boundary           ALERT 4/50
  CONTRACT_POS_boundary_plus_prior ALERT 5/50
CONTRACT-VIOLATING (want CONTRACT_INVALID high AND ALERT <=3/50):
  VIOLATION_cov_confound          INVALID 50/50 | ALERT 0/50   (medAUC 0.78)
  VIOLATION_cov_plus_prior        INVALID 50/50 | ALERT 0/50   (medAUC 0.78, the B7.1 killer)
  VIOLATION_no_within_block_rand  INVALID 50/50 | ALERT 0/50   (no support blocks)
  VIOLATION_borderline_confound   INVALID  4/50 | ALERT 0/50   (medAUC 0.586, 46 slip tau; still no false-alert)
  VIOLATION_lowvar_confound       INVALID 33/50 | ALERT 0/50   (medAUC 0.65, 17 slip tau; still no false-alert)
```

Screen: violations refused ✓, POS>0 ✓, but `contract_nulls_le3` **FAILS** (prior_only=5) and POS not strong →
self-reported verdict **MIXED**.

## Disclosed limitations (folded in from the result red-team — do NOT smooth over)

1. **HIGH — POS power does not yet exceed the collider-inflated null floor.** At the boundary effect size, POS power
   (POS_boundary 4/50 = 8%) does **not** exceed the worst null over-alert (prior_only 5/50 = 10%); Clopper-Pearson CIs
   fully overlap ([2.2,19.2]% vs [3.3,21.8]%). So at this effect size + n_eligible=30, the test **cannot yet separate a
   weak genuine concept from the prior-collider artifact**. "Safe-but-weak" is true only with this asterisk.
2. **prior_only over-alert 5/50 = 10% = approximate-exactness (collider), not a bug.** `fix_class_margins=True`
   stratifies the exact permutation by (Block, Y); Y depends on C (logit shift ±0.62) *and* on Z, so Y is a
   post-treatment collider — conditioning on it leaves residual within-(block,Y) C–Z dependence, left-shifting T. The
   "exact" null is only *approximately* exact for prior-bearing worlds. This is a B8.1 design target, not a p-value to
   retune.
3. **Stress-design gap.** No tested world combines a *quiet* confound that PASSES τ (like borderline AUC 0.586) *with* a
   C→Y prior effect. Both slip-through worlds (borderline, lowvar) have Y=draw(p0) — no concept — so "ALERT 0 on
   slip-through" is partly "nothing to detect," not proven robustness in the genuinely dangerous quiet-confound+C→Y cell
   (untested → B8.1 stress requirement).
4. **Emulator, not validation.** Lee2019 SM16 geometry is a realistic Z/subject substrate; the worlds are semi-synthetic
   injections. This is NOT a real randomized-audit dataset. n=50/world, single base seed → wide CIs on every rate.

## Verdict

**B8.0 = information-contract direction VALIDATED ON EMULATOR (MIXED).** Requiring a known randomization contract + exact
null **controls** the mixed covariate+prior cell that structurally broke the observational line and **refuses** confounded
data. It is neither a clean pass nor a failure. **NO tag.** NEXT (reviewer decision): B8.1 = class-balanced randomization
contract (move class/prior balance *into* the contract to lower the collider floor) + audit/label power budget (raise POS
*above* the floor so a weak concept separates from the artifact), or genuinely randomized-audit data. NOT another
witness/variant, NOT threshold tuning, NOT a new statistic/encoder.

Provenance: seeds `400e6 + world_index*1e6 + cohort`, all 600 unique, zero intersection with all prior CSC runs. Isolation:
`check_contract` takes no Y (contract-first); exact within-(block,class) permutation null with no fitted Y|Z or C|Z
propensity; alert depends only on p_meanT/p_stud/estimability/eligibility (no oracle/router/observed_T/B7-witness). See
`b8_stage0_redteam_checks.json`, `b8_stage0_manifest.json`, `notes/b8_information_contract.md`. Related:
`b7_stage1_full_replay/`, `b6_condition_randomization/`.
