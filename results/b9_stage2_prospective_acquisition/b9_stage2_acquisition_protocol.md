# B9.2 prospective randomized-audit acquisition cohort — protocol (development-only)

Reviewer-authorized 2026-07-07 after B9.1A showed the pipeline runs but a 6–10-subject cohort **cannot** produce alert-level
evidence under the unchanged `min_confirm_pairs=20` gate. **B9.2 targets a ≥20-subject cohort so alert-level null behavior
and power become assessable.** This is a **contract-feasibility + contract-valid null audit pilot, NOT full clinical
validation, NOT deployment, NOT a biological concept claim.** No real EEG data has been collected.

> **Governance.** Real human-EEG collection is gated on the institution's ethics/governance approval, informed consent,
> data-handling/privacy process, and lab safety/recording approval **before any recording**. This document is study design,
> not legal/IRB advice; the institution's process controls collection. Status is tracked (non-PII) in
> `b9_stage2_governance_status.md`.

## Cohort size

- **Enrollment slots:** 30 (pre-declared).
- **Target analyzable:** ≥ 24. **Hard minimum for alert-level analysis:** ≥ 20 (= `min_confirm_pairs`, **UNCHANGED**).
- **Rationale:** B9.1A verified that at n=8 the exact null runs and p-values can be ≤0.025, yet `size_ok` (n_eligible<20)
  correctly blocks any alert. A ≥20-eligible cohort is required to test alert-level behavior; 30 slots reserve for attrition.

## The minimal 2×2 factorial contract (unchanged from B9.1A)

- **Factors:** `C ∈ {0,1}` (audit condition), `Y_design ∈ {0,1}` (pre-assignment cue / intended class).
- **Unit:** `subject × microblock`. Within each microblock: equal counts for the 4 `(C, Y_design)` cells (quadruplet × R=3),
  randomized order from the pre-generated table.
- **Per subject:** 10 microblocks × 12 = **120 trials**; **planned total 30 × 120 = 3600**.
- `Y_design` is the **cue / intended class / stimulus class — never an observed or generated post-hoc `Y`.**

## Assignment table + adherence (trial_id join)

- `b9_stage2_assignment_table.csv` (+ `.sha256`), manifest attests `generated_before_recording=True`,
  `Y_design_pre_assignment=True`. Columns `subject_id, microblock_id, trial_id, C, Y_design`.
- **Adherence is checked by joining executed rows to the pinned table on `trial_id`** (the round-order-independent key the
  B9.0 red-team recommended). Every recorded trial must match its pinned `(C, Y_design, subject, microblock)` row.
- **Attrition** (dropped trials/subjects) is a *subset* of the pinned table: if it preserves within-`(subject × microblock ×
  Y_design)` balance → **reduced support** (`INSUFFICIENT_LABELS_OR_SUPPORT` if below threshold); if it induces imbalance or
  a prior shift `P(Y_design|C)` beyond tolerance → **`CONTRACT_INVALID_OR_OUT_OF_ESTIMAND`**. Attrition is never used to
  post-hoc rebalance.

## Early operations-only checkpoint (6–10 subjects)

An early checkpoint may inspect **ONLY logistics** — assignment adherence, trial attrition, artifact-rejection rate, per-cell
valid counts, support / sampler feasibility, contract-invalid reasons, recording runtime, participant burden (see
`operations_checkpoint.checkpoint_report`, which structurally never computes a p-value or alert). It **MUST NOT** view or act
on: `B9_ALERT`, exact-null p-values, `observed_T`, `T_z`, injected-positive results, or condition-specific scientific trends.
The checkpoint may pause acquisition for **logistics / support failure** only; it may **never** change the protocol based on
signal (weak or strong).

## Stop rule (NOT p-dependent)

Stop acquisition after **≥ 24 analyzable subjects OR all 30 slots exhausted**. "Analyzable" is fixed by the pre-declared
quality/support criteria (`b9_stage2_quality_criteria.json`) — never by p-values or effect sizes.

## Analyses (pre-registered; see `b9_stage2_preanalysis_protocol.json`)

- **A. Contract-valid null audit** on the real acquired data — the main question. **PENDING_ACQUISITION** (governance-gated).
  Because this is a single-cohort pilot, it is **not** reported as a 300-cohort CP estimate.
- **B. Forced-violation validator tests** on corrupted copies of the manifest/table (missing table, flipped C, shuffled
  `Y_design`, microblock imbalance, condition-lock, subject/microblock tuple mismatch, attrition-induced imbalance) → all
  must refuse (`CONTRACT_INVALID_OR_OUT_OF_ESTIMAND`) before any p-value. *(Verified here on synthetic Z.)*
- **C. Optional semi-synthetic positive** on the acquired real Z substrate — **diagnostic / power only, NOT biological.**

## Success screen (development; B9.2 is a pilot, not final validation)

- **Contract execution:** ≥ 90% recorded trials match the assignment table (before exclusions).
- **Analyzability:** ≥ 20 analyzable subjects (≥ 24 preferred).
- **Support:** ≥ 80% analyzable subjects satisfy per-cell support.
- **Validator:** 0 forced-violation cases produce `B9_ALERT`.
- **Contract-valid null audit:** no `B9_CONCEPT_ALERT` preferred; if an alert occurs → **stop and diagnose** the exact null /
  acquisition contract (not the gate).
- **Optional synthetic positive:** > 0 alert preferred (diagnostic).
- **If only 12–18 analyzable:** `INSUFFICIENT_LABELS_OR_SUPPORT` → **acquisition-budget redesign**, do **not** relax
  `min_confirm_pairs`.

## Not authorized

6–10-subject-only scientific pilot; gate relaxation / `min_confirm_pairs` change; p/mean-T/threshold recalibration;
feature/montage/statistic changes; post-hoc balancing on observed Y; case-control selection using Z; Lee2019 as validation;
existing-dataset route as primary; paper writing. A bounded B9.1B eligibility scan may be authorized later only for datasets
with genuine **pre-recording** `C × Y_design` provenance (most will fail).
