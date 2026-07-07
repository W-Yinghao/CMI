# B9.1A prospective randomized-audit pilot — PRE-ACQUISITION (diagnostic-only, NO scientific/biological claim)

Reviewer-authorized 2026-07-07 after B9.0 (contract machinery). This is the **pre-acquisition** stage of a prospective
randomized-audit pilot: a **frozen pre-registration** + a **synthetic-Z verification** of the B9.0 hardened pipeline.
**NO real data has been collected. NO scientific/biological/power claim is made. This is NOT validation.**

> **Governance.** Actual human-EEG collection is gated on the institution's local governance/ethics process **before any
> recording**. This is study design, not legal/IRB advice; the institution's process controls collection.

## Contents

- `b9_stage1_acquisition_protocol.md` — the pilot study design (2×2 `C × Y_design` contract; 8 subjects × 10 microblocks ×
  R=3; exclusion/attrition rules; governance gating). **The entry document.**
- `b9_stage1_assignment_table.csv` (+ `.sha256`) — the **pilot assignment table, generated BEFORE recording and hash-pinned**
  — the crux provenance artifact. Columns `subject_id, microblock_id, trial_id, C, Y_design`; 960 trials; `C × Y_design`
  exactly balanced within every subject × microblock.
- `b9_stage1_manifest.json` — attests `generated_before_recording=True`, `Y_design_pre_assignment=True` (attestation *floor*;
  data-level provenance is unverifiable from data and is guaranteed by the B9.1 acquisition protocol).
- `b9_stage1_preanalysis_protocol.json` — the pre-registered analyses A/B/C, success screen, states.
- `b9_stage1_rows.jsonl`, `b9_stage1_tables.json`, `b9_stage1_contract_checks.json` — the synthetic-Z pipeline-verification
  results (machinery only).
- `b9_stage1_redteam_checks.json` — design red-team (`w91e6s0py`, both lenses PASS).

## Analyses (pre-registered)

- **A. Contract-valid null audit** on real acquired data — **PENDING_ACQUISITION** (governance-gated; **not run**).
- **B. Forced-violation validator tests** (synthetic Z, on corrupted copies of the manifest/table): all 7 cases —
  flipped C, shuffled `Y_design`, missing table, condition-lock, microblock imbalance, corrupt pinned hash, post-hoc
  manifest — **refuse before any p-value** (`CONTRACT_INVALID_OR_OUT_OF_ESTIMAND`, `ran_test=False`). Verifies the B9.0
  hardened validator stays hardened in the pilot pipeline.
- **C. Semi-synthetic diagnostic / power-sizing** (synthetic Z following the locked table; **NOT biological**): at pilot
  n=8 → `NO_ACTIONABLE` (size gate); a power-sizing **simulation** at n=24 → concept `B9_CONCEPT_ALERT`, null `NO_ACTIONABLE`.

## Size-gate finding (disclosed design consideration, NOT a gate change)

The B9.0 alert conjunction requires `n_eligible ≥ min_confirm_pairs = 20` **subjects**. The authorized pilot (6–10) is below
that, so **the pilot cannot emit `B9_CONCEPT_ALERT` regardless of signal** — verified: at n=8 the exact null runs and
`p_meanT=0.005`, `p_stud=0.015` (both ≤0.025), and `size_ok` (8<20) is the *sole* unmet alert condition. Consequences: the
pilot verifies contract execution + exact-null-runs + violations-refuse, its "no false alert" is **size-gate-trivial**, and
**alert-level power AND null-control require a ≥20-subject cohort** (shown here only by the synthetic n=24 simulation).
`min_confirm_pairs` is **UNCHANGED** (no gate tuning). Reviewer decision: run the 6–10 feasibility pilot as-is (alert-level
deferred), or scope a ≥20-subject cohort for alert-level claims.

## Next (reviewer decision, NOT authorized)

After governance approval, acquire the pilot cohort under the pinned table and run analysis A; for alert-level claims a
≥20-subject cohort is required. **NOT authorized:** existing-dataset validation as primary, Lee2019 as validation, post-hoc
balancing on observed Y, mean-T/p recalibration, new statistic/feature, B8.4, B7 variants, paper writing. Related:
`csc/b9/CONTRACT.md`, `../b9_stage0_contract_design/` (B9.0), `../b8_stage3_label_balanced_contract/` (B8.3).
