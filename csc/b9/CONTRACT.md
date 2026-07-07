# B9 prospective randomized-audit contract (development-only, NO scientific claim)

Reviewer-authorized 2026-07-07 after B8.3 (INSUFFICIENT): the whole B8 emulator line proved that **post-hoc label
balancing on an observed/generated label Y cannot remove the collider** — balancing on Y (a common effect of C and Z)
leaves a second-order residual (selection-intensity asymmetry + within-Y C-Z). B9 stops repairing prior after the fact
and moves label/class balance and condition randomization **into the acquisition contract**.

B9.0 is **design + schema + exact-null + state-machine + dry-run ONLY**. It makes **NO scientific positive claim**, NO
power claim, and is NOT validation. Lee2019 may later serve as a code dry-run substrate but never as B9 validation.

## Estimand (narrow, falsifiable)

**INSIDE:** under a pre-randomized, class-balanced, within-block audit contract, detect **condition-specific
boundary / interaction evidence** (a C×Z interaction in decoding the pre-assigned class `Y_design`) **beyond known
randomization variation**.

**OUTSIDE (→ `CONTRACT_INVALID_OR_OUT_OF_ESTIMAND`):**
- natural-prevalence deployment concept certification (all P(Y|Z,C) change at natural prior);
- any dataset where `P(Y_design | C)` differs (a genuine prior shift) — B9 does NOT try to control it, it **refuses**;
- covariate confounding that violates the balance audit;
- condition assignment that is not randomized/counterbalanced (e.g. a session label).

B8.3 already proved post-hoc repair of prior leaves a collider; therefore prior imbalance **invalidates the contract**,
it is not a "valid null to control."

## The contract (acquisition-time, hash-pinned BEFORE recording)

- **Unit:** `subject × microblock × design_class`.
- **`Y_design`** = a **pre-assignment cue / task / stimulus class** (what the subject is instructed to do), fixed BEFORE
  recording. It is **NOT** an observed or generated post-hoc label. (This is the single difference from B8.3.)
- **`C`** = condition, **randomized / counterbalanced** within `subject × microblock`, in the same session or a tightly
  matched block. NOT a session/subject label.
- **Balance:** within each `subject × microblock`, the `C × Y_design` cell counts are **equal (or a predeclared count)**.
  Recommended microblock = randomized-order repeats of the quadruplet `{(C=0,Y=0),(C=0,Y=1),(C=1,Y=0),(C=1,Y=1)}`, ×R.
- **Timing:** the assignment table is generated + hashed **before recording**; `Y_design` and `C` provenance are
  pre-recording. Prior shift or condition confounding is then a question of **whether the contract was met**, not an
  analysis-time repair.

**Enforcement (B9.0 code) vs attestation (B9.1 protocol).** The validator ENFORCES what is checkable from
`(C, Y_design, subject, microblock, table)`: (i) a valid hash-pinned table exists; (ii) **the manifest attests
`generated_before_recording=True` AND `Y_design_pre_assignment=True`** (a boolean *floor* — data-level provenance, i.e.
whether `Y_design` was *truly* the pre-assigned cue and the table *truly* predated recording, is **inherently unverifiable**
from the data and must be guaranteed by the B9.1 acquisition protocol); (iii) **the executed full pre-registered tuple
`(C, Y_design, subject, microblock)` FOLLOWS the registered table** row-for-row (`adherence = 1.0` — any deviation means
the pre-registration is not binding, an anti-p-hacking requirement — the analyst may post-hoc relabel *neither* `C` *nor*
`Y_design`; since the exact null holds `Y_design` fixed, a `Y_design` relabel is the same class of hole as a `C` relabel);
(iv) executed balance / no-prior-shift / randomization support. **Row-order convention:** the executed inventory and the
pinned table must be delivered in the same `(subject, microblock, trial)` order (positional alignment); B9.1 should join on
a registered per-trial id. B9 without this binding would be **decorative** — it is the single property that distinguishes B9 from the B8 line.

## Exact randomization null (no fitted models)

B9 does **NOT** estimate `Y|Z` or `C|Z,S`, and does **NOT** case-control-select on an observed `Y`. It:
- **Fixes:** `Z`, `Y_design`, `subject_id`, `microblock_id`, the assignment-table support, and the contract-dictated
  `C × Y_design` counts.
- **Resamples:** `C*` only within the **predeclared randomization set** — a uniform permutation of `C` within each
  `(subject, microblock, Y_design)` stratum (this preserves the contract's `C × Y_design` balance exactly). Because
  `Y_design` is **pre-assignment** (⟂ C, not a collider), this permutation is **collider-free CONDITIONAL on genuine
  within-stratum conditional randomization of the executed trials.** The Z-blind count-balance + `P(Y_design|C)` checks
  are *necessary but not sufficient*: a Z-dependent, balance-and-prior-preserving differential attrition/artifact-rejection
  within a stratum would re-open a C–Z dependence yet pass the Z-blind validator. That residual is out of reach of a
  Z-blind check by design and must be excluded by the **B9.1 acquisition protocol / provenance**, not by this validator.
- **Recomputes:** the **same** contrast `T(Y_design, Z, C*)` (byte-reused B3 cross-fit interaction test).

## States (fail-closed, contract-FIRST)

- `B9_CONCEPT_ALERT` — contract valid + support adequate + **`estimable` (n_exact_invalid ≤ 0.20·n_boot) AND
  `p_meanT ≤ 0.025` AND `p_stud ≤ 0.025` AND `n_eligible ≥ 20`** (the exact alert conjunction; B9.0 does NOT include the
  B3 subject-consistency LCB>0 gate — harmless here since B9.0 makes no positive claim, but it must be reconsidered before
  any B9.1 confirmatory use).
- `NO_ACTIONABLE_CONCEPT_EVIDENCE` — contract valid + support adequate + not significant.
- `CONTRACT_INVALID_OR_OUT_OF_ESTIMAND` — missing/invalid assignment table; `C × Y_design` imbalance beyond tolerance;
  condition not randomized/counterbalanced (condition-lock / session confound); prior shift from attrition/noncompliance;
  covariate confounding violates balance; or a natural-prevalence target requested. **Decided BEFORE any p-value.**
- `INSUFFICIENT_LABELS_OR_SUPPORT` — too few trials per `subject × block × class × condition`.
- `SAMPLER_INVALID` — exact null infeasible.

Never emit `NO_CONCEPT`. The contract validator uses ONLY `(C, Y_design, subject, microblock, assignment table)` — it
**never reads `Z` or the test statistic `T`**.

## Hard stops (any of these ⇒ not science; a bug to fix)

- assignment table generated **after** data/labels exist;
- `Y_design` is actually an observed/generated post-hoc `Y`;
- `C` is a session label rather than a randomized/counterbalanced condition;
- the validator uses `Z` or `T`;
- prior imbalance is **repaired** after observation instead of **invalidating** the contract;
- an invalid world reaches p-value testing;
- state counts are not disjoint;
- randomization support too small but the certifier still alerts.

## B9.0 deliverables + what is NOT authorized

Code: `csc/b9/` (this dir). Package: `csc/results/b9_stage0_contract_design/`. B9.0 = design + dry-run + design red-team +
diagnostic-only commit. **NOT authorized:** B8.4, mean-T/p recalibration, selector/statistic/feature changes, power
frontier, Lee2019-as-validation, confirmatory tag, paper writing. **B9.1** (real prospective acquisition OR an existing
genuinely-pre-randomized dataset) is a separate future authorization.
