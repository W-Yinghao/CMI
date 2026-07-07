# B9.1A prospective randomized-audit acquisition pilot — protocol (development-only)

Reviewer-authorized 2026-07-07 after B9.0 (contract machinery). **This is a prospective contract-feasibility + null-control
pilot, NOT a validation and NOT a method-success claim.** No biological/concept claim is made. The pilot exists to move from
the B8 emulator boundary to a real acquisition contract.

> **Governance.** Actual human-EEG collection MUST be approved through the institution's local governance / ethics process
> BEFORE any recording. This document is a study design, not legal/IRB advice; the institution's process controls collection.
> **No data has been collected.** The artifacts here are the pre-registration (frozen assignment table + pre-analysis
> protocol) and a synthetic-Z verification of the pipeline.

## What the pilot answers (feasibility + null-control, not power)

1. Can a hash-pinned assignment table be executed faithfully in real acquisition (adherence to the pinned schedule)?
2. Is `Y_design` genuinely a **pre-assignment cue**, not an observed/generated post-hoc label?
3. Is within-`(subject × microblock × Y_design)` condition randomization executable?
4. Does the exact randomization null run on real data (`ran_test = True`)?
5. Do contract-invalid conditions **fail-closed** (refuse before any p-value)?
6. Under a real contract-valid null, is there no obvious false alert? *(See the size-gate finding — at pilot n this is
   size-gate-trivial, not a strong test.)*

## The minimal 2×2 factorial contract

- **Factors:** `C ∈ {0,1}` (audit condition), `Y_design ∈ {0,1}` (cue / intended class, **pre-assignment**).
- **Unit:** `subject × microblock`.
- **Balance:** within each microblock, equal counts for `(C=0,Y=0), (C=0,Y=1), (C=1,Y=0), (C=1,Y=1)`.
- **Order:** randomized within each microblock from the pre-generated assignment table.

**Locked pilot dimensions:** **8 subjects × 10 microblocks × R=3** trials per 2×2 cell per microblock → 12 trials/microblock,
**120 trials/subject, 960 total** (within the reviewer's 64–192/subject range). Lower the count if acquisition burden is high;
the purpose is feasibility + null behavior, not final power.

## Assignment table (the crux provenance artifact)

- **Generated BEFORE recording** and **hash-pinned** — this is the single property that distinguishes B9 from the B8 line
  and is essentially unrecoverable from ordinary historical EEG.
- Columns: `subject_id, microblock_id, trial_id, C, Y_design` (see `b9_stage1_assignment_table.csv`, sha in `.sha256`,
  attestation in `b9_stage1_manifest.json` with `generated_before_recording=True`, `Y_design_pre_assignment=True`).
- The executed acquisition must **adhere to the pinned table row-for-row, joined on `trial_id`**; any mismatch of the full
  tuple `(C, Y_design, subject, microblock)` → `CONTRACT_INVALID_OR_OUT_OF_ESTIMAND` (the B9.0 hardened validator).
- **`Y_design` is the cue / intended class / stimulus class — never an observed or generated post-hoc `Y`.**

## Exclusions / attrition (pre-declared; NOT post-hoc rebalancing)

Behavioral failures, missed trials, artifacts, and noncompliance are handled as **attrition / predefined exclusion**, never
by rebalancing after seeing outcomes. Pre-declared in `b9_stage1_preanalysis_protocol.json`: artifact-rejection rule,
minimum valid trials per `subject × microblock × C × Y_design`, whether rejected trials **break the contract** (if they
induce imbalance / prior shift beyond tolerance → `CONTRACT_INVALID_OR_OUT_OF_ESTIMAND`) or merely **reduce support**
(→ `INSUFFICIENT_LABELS_OR_SUPPORT`). Mostly-`INSUFFICIENT_SUPPORT` is **not a method failure** — it means the acquisition
budget is too small.

## States (B9.0 state machine, unchanged)

`B9_CONCEPT_ALERT` / `NO_ACTIONABLE_CONCEPT_EVIDENCE` / `CONTRACT_INVALID_OR_OUT_OF_ESTIMAND` /
`INSUFFICIENT_LABELS_OR_SUPPORT` / `SAMPLER_INVALID`. Never `NO_CONCEPT`.

## SIZE-GATE FINDING (disclosed design consideration, NOT a gate change)

The B9.0 alert conjunction requires `n_eligible ≥ min_confirm_pairs = 20` **subjects**. The authorized pilot size (6–10
subjects) is **below** that, so **the pilot cannot emit `B9_CONCEPT_ALERT` regardless of signal.** Consequences, stated
honestly:
- The pilot verifies **contract execution + exact-null-runs + violations-refuse**, and its "no false alert" is
  **size-gate-trivial** (blocked by `n_eligible < 20`), not a strong null test.
- **Alert-level null-control AND alert-level power both require a cohort of ≥ 20 eligible subjects** (shown here only by a
  synthetic power-sizing simulation in `b9_stage1_tables.json`).
- `min_confirm_pairs` is **UNCHANGED** (no gate tuning). This is a reviewer decision: run the 6–10 feasibility pilot as-is
  (alert-level deferred), or scope a ≥20-subject cohort for alert-level claims.

## Analyses (pre-declared) — see `b9_stage1_preanalysis_protocol.json`

- **A. Contract-valid null audit** on the real acquired data — **PENDING_ACQUISITION** (governance-gated; not run here).
- **B. Forced-violation validator tests** on corrupted **copies of the manifest/table** (not raw data): flipped C, shuffled
  `Y_design`, missing table, condition-lock, imbalance, corrupt pin, post-hoc manifest → all must refuse before any p-value.
  *(Run here on the pilot's actual table + synthetic Z — machinery check.)*
- **C. Optional semi-synthetic positive** on synthetic Z following the locked table — a **diagnostic / power-sizing** run,
  explicitly **not a biological concept claim.**

## Not authorized

Existing-dataset validation as primary; Lee2019 as validation; post-hoc balancing on observed Y; case-control selector using
Z; mean-T/p recalibration; new statistic/feature/montage; B8.4; B7 variants; paper writing. A bounded B9.1B eligibility scan
of existing datasets may be authorized later, but ONLY if a dataset carries genuine **pre-recording** `C × Y_design`
assignment provenance (most will fail this).
