# CSC B8.1 class-balanced randomized-audit contract — design (development-only)

```
Scope: B8.1 class-balanced randomized-audit contract | design_class-stratified exact null + hard provenance gate + Lee2019 emulator
  development-only | NOT confirmatory | NOT deployable | NOT validation (Lee2019 is a real-geometry emulator, not a real randomized audit) | NO tag
  reviewer-authorized 2026-07-06 after B8.0 MIXED (controls the mixed cov+prior cell but prior-only collider floor 5/50 + weak POS)
```

**Why B8.1 (targets exactly the two B8.0 limitations, nothing else).** B8.0 stratified the "exact" null by the
OBSERVED audit label Y (`resample_C_exact` key `(Block, Y)`). Y depends on C (prior logit shift) AND on Z, so Y is a
**post-treatment collider** — conditioning on it left residual within-(block,Y) C–Z dependence → `CONTRACT_NULL_prior_only`
over-alerted 5/50, and that collider floor was not separated from the genuine POS power (4/50). B8.1 moves class/prior
balance **into the randomization contract** instead of conditioning on observed Y after the fact.

**The single causal change.** design_class `Dc` = the **real cued class `y0`** — a PRE-ASSIGNMENT / pre-outcome design
variable (the MI cue precedes any synthetic condition assignment; `Dc ⟂ C` by the randomization; NOT the observed
post-hoc Y). Then:
1. condition `C` is **balanced-randomized within `(subject × block × Dc)` strata** (class-balanced randomization);
2. the EXACT null resamples `C*` **within `(block, Dc)` strata** — a pre-assignment stratifier → **no collider**;
3. everything else — the B3 contrast `T`, the studentized gate, estimability, `min_confirm_pairs`, the feature cache —
   is **byte-unchanged** from B8.0. No statistic change, no feature/montage change, no p-value retuning.

**Hard provenance gate (primary) + AUC diagnostic (secondary).** B8.0's contract check was AUC(C~Z)≤τ + support —
which QUIET confounds slipped (B8.0 stress-design gap). B8.1 requires a **registered randomization schedule** generated
BEFORE any label/concept injection and hash-pinned, and the certifier verifies:
- **H1** a schedule exists; **H2** its hash matches the pin (integrity); **H3** the executed `C` FOLLOWS the schedule
  (`C == C_table`, adherence); **H4** within-`(block,Dc)` support ≥ 8 strata; **H5** the schedule is balanced within
  `(block,Dc)` (`|n_HI−n_LO| ≤ 1`).
- **D1 (secondary, fail-closed):** block-adjusted `AUC(C~Z) ≤ τ=0.60`, reported for EVERY cohort.

`valid = H1..H5 ∧ D1`. Any failure → `CONTRACT_INVALID_OR_UNIDENTIFIABLE` with a reason from
`{missing_randomization_table, assignment_hash_mismatch, assignment_not_following_schedule, insufficient_support,
condition_lock_no_within_stratum, balance_diagnostic_fail, block_confounding_auc}`. **Provenance is the primary gate:**
a quiet confound (small schedule deviation, low AUC) is REFUSED by **H3 adherence** even though **D1 AUC would pass** —
this is the point, and closes the B8.0 stress gap. Contract-FIRST: an invalid contract refuses BEFORE `T` is computed,
so even a genuine concept under an invalid contract (`VIOLATION_quiet_cov_plus_concept`) is REFUSED, never alerted.

> Honest note on the provenance gate: schedule-adherence (H3) is a **pre-registration verification** — it checks the
> executed assignment matches a hash-pinned schedule generated before injection. It is NOT claimed to "discover"
> confounding from observational data alone (that is what B3→B7 failed at). The information-contract value is precisely
> that we REQUIRE + VERIFY a known randomization rather than estimate a null post-hoc. The empirical contributions the
> canary tests are (i) the collider fix on CONTRACT worlds and (ii) that H3 refuses the quiet cells D1 misses.
>
> **Design-red-team disclosures (wtp17jn53, 3/3 MINOR_ISSUE, no blockers — folded in before compute):**
> - **H1/H2/H5 are non-discriminative by construction in this emulator** (H2 hash-integrity and H5 schedule-balance
>   always pass; no world corrupts the pinned table). The canary exercises only **H3 (adherence)** — the sole novel
>   discriminator carrying every violation refusal — plus **H4 (support**, fires on `condition_lock`) and **D1 (AUC,
>   secondary)**. Read the result as an **H3/H4 test, not a "5-gate" validation.**
> - **H3 is zero-tolerance** (`match < 1.0` refuses). In the emulator this is a clean 0/1 readout of the build branch;
>   robustness to *noisy/graded* real-world adherence (a real audit with a few mis-recorded trials) is **out of scope**.
>   `_schedule_hash` pins per-`(block,Dc)` **counts**, not the per-trial assignment — H3 is the per-trial gate; H2 must
>   not later be cited as per-trial integrity.
> - **2nd-order mean-T residual:** permuting `C` with `Y` held fixed breaks the `C→Y` prior main-effect link, so the
>   exact null is exact only for the **sharp interaction-null** (h0 absorbs the main effect). `prior_only` safety
>   therefore leans on the **studentized AND-gate**, not the mean-T gate. The merge reports mean-T-alone vs both-gate
>   co-fire (section E) so a mean-T drift toward 0.025 is visible before it converts to a both-gate alert.
> - **POS is expected weak on this substrate** (a 4-seed check gives near-identical near-zero T_obs under B8.1 and B8.0
>   nulls) — the `(block,Dc)` stratifier does **not** absorb the concept; weak POS is a pre-existing studentized-gate
>   property (B8.0 was 4/50), covered by the screen's safe-but-weak / powerless branches — NOT a B8.1 regression.

**Estimand (unchanged from B8.0, narrow):** condition-specific boundary/conditional-structure evidence BEYOND
label-prior shift, under a randomized/counterbalanced paired audit contract. Prior shift stays a nuisance.

**States:** `B8_CONCEPT_ALERT` / `NO_ACTIONABLE_CONCEPT_EVIDENCE` / `CONTRACT_INVALID_OR_UNIDENTIFIABLE` /
`INSUFFICIENT_LABELS` / `SAMPLER_INVALID`. Never `NO_CONCEPT`.

**Emulator (`build_b8_1_cohort`, Lee2019 SM16 geometry; ORDER enforced: blocks + Dc + registered schedule + hash BEFORE
any injection).** 12 worlds:
- CONTRACT (execute `C == C_table`): `CONTRACT_NULL_balanced`, `CONTRACT_NULL_prior_only` (**primary target**),
  `CONTRACT_NULL_cov_plus_prior` (must **stay** controlled), `CONTRACT_random_label`, `CONTRACT_POS_boundary`,
  `CONTRACT_POS_boundary_plus_prior`.
- VIOLATION (executed assignment DEVIATES from the schedule → REFUSE): `VIOLATION_cov_session`,
  `VIOLATION_prior_shift`, `VIOLATION_cov_plus_prior` (the B7.1 killer), `VIOLATION_condition_lock`,
  `VIOLATION_quiet_cov_no_concept` (quiet, low AUC, no concept), `VIOLATION_quiet_cov_plus_concept` (quiet + REAL
  concept — must REFUSE via provenance, NOT alert; closes the B8.0 stress gap).

**Canary (after design red-team clean + protocol commit/hash):** n=50/world, SM16, same B3 `T`, exact `(block,Dc)`
randomization null, NO feature change, NO learned encoder. **Screen (pre-registered):**
- contract-valid nulls `B8_ALERT ≤ 3/50` each — **`CONTRACT_NULL_prior_only` is the primary target** (was 5/50);
  `CONTRACT_NULL_cov_plus_prior` must **stay ≤ 3/50** (B8.0 was 1/50); `CONTRACT_random_label ≤ 1/50` pref.
- violation worlds: `CONTRACT_INVALID` high AND `B8_ALERT ≤ 3/50` each. For `quiet_cov_plus_concept`, success =
  REFUSE due to invalid contract, NOT alert-because-concept-exists.
- POS: `CONTRACT_POS_boundary B8_ALERT > 0` (≥5/50 strong); `CONTRACT_POS_boundary_plus_prior` reported separately.
- Interpretation: prior-only controlled + mixed controlled + POS≥5 → direction strong (discuss budget frontier LATER,
  NOT authorized). prior-only controlled + mixed controlled + POS weak → safe-but-weak (next step = audit-budget
  frontier, separate authorization). prior-only still high → class-balanced contract did NOT fix the collider (inspect
  design, do NOT retune p). mixed fails → direction unstable, STOP + diagnose. violation alert → provenance/validator
  bug, fix before any science claim.

**Red-team battery (design, before compute + result, after canary):** contract provenance (schedule generated before
labels/injections; `Dc` is pre-assignment, NOT observed/generated post-hoc Y; hash-pinned; no post-hoc selection); exact
null (`C*` only from `(block,Dc)` randomization; per-stratum counts preserved; no fitted Y|Z or C|Z propensity);
state logic (`CONTRACT_INVALID` decided before p-values; disjoint states sum to n; fail-closed); contamination (no
oracle/B7-witness/router/observed_T threshold; no feature/montage change; no statistic change); reporting (valid vs
violation separated; prior-only reported explicitly; no pooled safety claim). **Hard stop if:** observed/generated Y is
used as a randomization stratum; contract-invalid worlds alert before provenance refusal; schedule generated after
outcome/injection; states don't sum; the statistic or feature cache changes.

**NOT authorized (this stage):** B8.2 n=300 scale-up, audit-budget/power frontier, new statistic, new feature family,
deep encoder, threshold tuning, p-value recalibration, B7 variants, real-data validation claim, paper writing.

Dev code: `realeeg_feas/realeeg_b8_1.py` (emulator + `check_contract_b8_1` + `resample_C_exact_b8_1` + `b8_1_certify`).
Package (after canary): `csc/results/b8_stage1_class_balanced_contract/`. Builds on [[csc-b8-information-contract]],
[[b8_information_contract]], [[b7_stage1_full_replay]].

---

## RESULT (post-run, 2026-07-07) — MEETS pre-registered targets on the emulator (MODEST, masking-dependent), NOT "strong", NOT a statistic-level collider fix. NO tag.

12 worlds × 50 = 600 cohorts, base 420e6, SM16. Result red-team `wvfdv8j1c`: accounting/isolation **PASS** (600 rows
reproduced, 0 mismatches, seeds disjoint, isolation clean, T byte-identical); science **MINOR_ISSUE** — the decision-level
screen is genuinely MET but the auto-label "STRONG"/"collider FIXED" **over-claimed** (caught + corrected).

**Decision-level screen MET:** all contract nulls ≤3/50 (`prior_only` **2/50**, `cov_plus_prior` **1/50** retained,
`balanced` 1, `random_label` 2); all 6 violations **CONTRACT_INVALID 50/50 + 0 alerts**; quiet stress cells **50/50 AUC≤τ
yet 50/50 provenance-refused**, 0 alerts incl `quiet_cov_plus_concept` (real concept, refused contract-first);
`POS_boundary` **8/50 ≥5**.

**But the honest reading (do NOT say strong/fixed):**
1. **Collider CONTROLLED (decision-level), not FIXED (statistic-level).** `prior_only` mean-T-**alone** is INTACT at 5/50
   (= B8.0's both-gate level; binom vs 0.025 p=0.008). The both-gate 2/50 holds only because the studentized AND-gate did
   not co-fire (imperfectly — 2 of 3 studentized fires coincide with mean-T, residual leaks into the 2 alerts). The 5→2
   drop is CI-overlapping (Fisher p=0.22). Safety on prior worlds leans on the conservative studentized gate.
2. **Aggregate null control FLAT:** pooled 7/200 (B8.0) → 6/200 (B8.1); `balanced` 0→1 and `random_label` 1→2 got worse;
   only `prior_only` moved = redistribution within noise.
3. **POS modest + statistically unchanged vs B8.0:** 8/50 = 16% (84% miss); separates from the *pooled* floor (p=0.002)
   but not the mean-T floor (5/50, p=0.28); 4→8 vs B8.0 is p=0.36 (not higher).
4. **Genuine:** mixed retained (1/50); the stress-gap closure (provenance H3 catches the quiet confound AUC misses). But
   violation refusal is **by construction** (schedule deviation → H3 → contract-first).

**NEXT (reviewer decision, NOT authorized):** recalibrate the mean-T gate on prior worlds (so control does not lean on the
studentized AND-gate); an audit/label budget frontier to lift POS above the *mean-T* floor (not just the pooled floor); a
multi-seed replication to resolve the CI-overlapping 5→2 / 4→8; or genuinely randomized-audit data. NOT threshold tuning,
NOT a new statistic, NOT a paper claim. Committed diagnostic-only; package `csc/results/b8_stage1_class_balanced_contract/`.

---

## B8.2 RESULT (multi-seed stability, 2026-07-07) — B8.1's decision-level stability FALSIFIED. NO tag.

Reviewer authorized B8.2 = multi-seed stability replication (NOT mean-T recalibration, NOT power frontier, NOT real data).
Engine BYTE-IDENTICAL to committed B8.1 (module sha `dae229e39d89c940`, worker `86f8dd0bb6da4ec9`); 6 fresh disjoint seed
blocks (bases 500/520/540/560/580/600e6) × 12 conditions × 50 = 3600 cohorts. Protocol pre-registered + committed `d2aaf5f`
(sha `62cf256e08d62551`) BEFORE the run. Package `csc/results/b8_stage2_multiseed_stability/`; note commit + results commit
separate; NO tag. Result red-team `w3u5q3x10`: accounting/isolation **PASS** (3600 reproduced, engine byte-identical, seeds
disjoint, both-gate recompute matches, no fabrication); science **MINOR_ISSUE** (negative genuine + honestly framed).

**FALSIFIED (Case C AND Case D):** B8.1's decision-level "meets targets" does NOT replicate. At n=300:
- `CONTRACT_NULL_prior_only` **18/300 = 6.0%** (CP95 [3.6,9.3]%, strictly > nominal 2.5%, binom p=6.6e-4) — **FAILS** ≤7/300.
- `CONTRACT_NULL_cov_plus_prior` **13/300 = 4.3%** (fails the 7/300 screen, binom p=0.041; 95% CI touches nominal) — **FAILS**.
- per-block prior_only [2,2,3,3,3,5], mixed [3,1,3,2,2,2] — fail across ALL 6 blocks, not one outlier.
- **B8.1's 2/50 & 1/50 were TYPICAL underpowered single-block draws** from a true ~4-6% rate (block 0 reproduced 2/50
  exactly); the n=50 screen could not resolve ~5% from 2.5%. B8.1 never had genuine control — it lacked the power to see it.
- The studentized both-gate is **anti-conservative on prior-bearing nulls** (clean balanced/random 5/300 ≈1.7% well-
  calibrated; prior/mixed 2-2.4× nominal). mean-T-alone 46/49 (~15%); studentized masks ~60-75% but leaks a seed-dependent
  residual. Exactly the B8.1 result-red-team prediction.

**Survivors (NOT falsified — the direction is not dead):** violations 0/300 alerts, 300/300 refused, every block 0 (but
**by construction** via H3); POS genuine-but-modest (POS_boundary 37/300 ≈12.3% ≥20; POS+prior 39/300 ≥15; Fisher p=0.010
vs the failing null, but CP95 overlaps prior_only at the margin — not a clean detector).

**NEXT (reviewer, NOT authorized):** contract redesign / narrow the estimand + diagnose contract construction (Case C+D) —
**NOT** mean-T gate recalibration (explicitly rejected; would be post-hoc null repair), NOT p-tuning, NOT a budget frontier
(power isn't the blocker; null control is). Builds on [[csc-b8-information-contract]].

---

## B8.3 RESULT (label-balanced case-control audit contract, 2026-07-07) — INSUFFICIENT: halves but does NOT control the prior-collider residual. NO tag.

Reviewer authorized B8.3 = contract redesign (NOT gate-tune) after B8.2's Case C+D: remove label-prior via the AUDIT
SAMPLING CONTRACT (a predeclared, Z-blind, deterministic case-control selector `A(C,Y,G,Block,seed)` that balances P(Y|C) by
construction; the exact null RE-APPLIES A under every randomized C*). mean-T-alone made a PRIMARY screen. Reuses B8.1 engine
byte-frozen; new code = selector + certifier. Narrowed estimand = label-balanced audit population. Protocol pre-registered +
committed `d99ab79` (module sha `fa59a341`) BEFORE run; design red-team `wrznv3lin` (no blockers). Phase B = 6 fresh disjoint
blocks (620-720e6) × 12 × 50 = 3600. Package `csc/results/b8_stage3_label_balanced_contract/`; note + results commits
separate; NO tag. Result red-team `w1urtx7hm`: accounting/isolation **PASS** (engine byte-identical, seeds disjoint, exact
C×Y balance verified, 13-field bit-for-bit re-run, no fabrication); science **MINOR_ISSUE** (INSUFFICIENT genuine + decisive).

**INSUFFICIENT (pre-registered):** the mean-T PRIMARY screen (≤7/300) FAILS —
- `CONTRACT_NULL_prior_only` mean-T-alone **28/300** (CP95 [6.3,13.2]%), both-gate 2/300.
- `CONTRACT_NULL_cov_plus_prior` mean-T-alone **22/300** (CP95 [4.7,10.9]%), both-gate 10/300.
- The sampling contract HALVED the residual (B8.2 46→28 Fisher p=0.034; 49→22 p=9e-4) — a GENUINE first-order-prior removal,
  worked as designed — but does NOT control it to nominal. **Do NOT read the studentized-masked both-gate 2/300 as success**
  (mean-T is primary precisely because the both-gate would falsely pass here).
- **Mechanism confirmed:** channel (a) selection-intensity asymmetry −390/−388 (prior/mixed) vs +2 (balanced), co-located with
  the breaches → the surviving second-order collider residual, not a code fault; channel (b) within-Y C-Z design-asserted.
- POS survives but `POS_boundary` **19/300 < 20** strong bar = POWER (smaller balanced sample; mean-T 93 shows signal intact,
  NOT absorption); POS+prior 30/300 ≥15. Violations refuse 300/300 (by construction, H3).

**NET:** the information-contract line (B8.0→B8.3) is pushed to its emulator limit: case-control-on-a-collider inherently
leaves a second-order residual count-balancing cannot remove. **NEXT (reviewer, NOT authorized):** B9 genuinely randomized
audit acquisition OR estimand-narrowing (declare prior-bearing worlds out of target) — NOT gate/mean-T recalibration, NOT
p-tuning, NOT selector/statistic changes to match intensities. Builds on [[csc-b8-information-contract]].
