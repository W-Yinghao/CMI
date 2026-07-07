# B9.2 prospective randomized-audit acquisition cohort — PRE-ACQUISITION (diagnostic-only, NO real data / NO biological claim)

Reviewer-authorized 2026-07-07 after B9.1A showed a 6–10-subject pilot cannot produce alert-level evidence under the
unchanged `min_confirm_pairs = 20` gate. **B9.2 scopes a ≥20-subject cohort** (target ≥24 analyzable, 30 enrollment slots)
with a 6–10-subject **operations-only** checkpoint. This is a **frozen pre-registration + a synthetic-Z verification of the
pipeline** — **NO real EEG data has been collected, NO scientific/biological/power claim is made, this is NOT validation.**

> **Governance.** Real human-EEG collection is gated on the institution's ethics/governance approval, consent, data-handling,
> and lab-safety process **before any recording** (see `b9_stage2_governance_status.md` — `status: not_started`). This is
> study design, not legal/IRB advice.

## Contents

- `b9_stage2_acquisition_protocol.md` — the study design (2×2 `C × Y_design`; 30 slots / ≥24 analyzable; trial_id-join
  adherence; operations checkpoint; stop rule; governance). **Entry document.**
- `b9_stage2_assignment_table.csv` (+ `.sha256`) — the **30-slot table, generated BEFORE recording, hash-pinned** (columns
  `subject_id, microblock_id, trial_id, C, Y_design`; 3600 planned trials; `C × Y_design` exactly balanced per subject×microblock).
- `b9_stage2_manifest.json`, `b9_stage2_subject_slot_manifest.json` (30 slots, stop rule), `b9_stage2_quality_criteria.json`
  (analyzable definition + attrition semantics), `b9_stage2_preanalysis_protocol.json`, `b9_stage2_governance_status.md`.
- `b9_stage2_rows.jsonl`, `b9_stage2_tables.json`, `b9_stage2_contract_checks.json`, `b9_stage2_forced_violation_checks.json`
  — the synthetic-Z pipeline-verification results (machinery only).
- `b9_stage2_redteam_checks.json` — design red-team (`wriifipum`, Lens A PASS / Lens B MINOR_ISSUE, byte-reproducible).

## Why ≥24 (the B9.1A size-gate finding, resolved the right way)

The B9.0 alert conjunction requires `n_eligible ≥ min_confirm_pairs = 20` subjects. B9.1A verified a 6–10 pilot **cannot**
alert (at n=8 the null runs, p's can be ≤0.025, but `size_ok` blocks). B9.2 **meets the gate with ≥24 analyzable rather than
relaxing it** — `min_confirm_pairs` is **UNCHANGED**. If only 12–18 turn out analyzable → `INSUFFICIENT_LABELS_OR_SUPPORT` →
acquisition-budget redesign, **not** a gate change.

## Pipeline verification (synthetic Z, machinery only — NOT biology)

- **B. Forced violations** (7 cases via the **trial_id-join** validator): missing table, flipped C, shuffled `Y_design`,
  tuple mismatch, microblock imbalance, condition-lock, attrition prior-shift → **all refuse before any p-value**.
- **A-synth. Contract-valid null audit at alert-capable n:** full n=30 → `NO_ACTIONABLE` (`n_eligible=30`, `p=0.995` — a
  **genuine, non-size-gate-trivial** null, *unlike* the size-gated 6–10 pilot); 20% whole-subject **attrition → 24
  analyzable** (balance preserved) → `NO_ACTIONABLE` (reduced support, **not** a violation).
- **C. Synthetic positive (power):** injected boundary → **ALERT at both n=24 and n=30** (`p=0.005`).
- **Operations checkpoint (6–10):** `checkpoint_report` structurally emits **only logistics** (adherence, attrition,
  support, sampler feasibility, contract-invalid reasons) and **never computes a p-value or alert** — a genuinely-adhering
  8-subject cohort reads `VALID`, a flipped-C one `INVALID`, with no scientific endpoint exposed.

## Analyses

- **A. Real contract-valid null audit** on the acquired ≥24 cohort — **PENDING_ACQUISITION** (governance-gated; not run;
  reported as a single-cohort pilot, not a 300-cohort CP estimate).
- **B / C** verified here on synthetic Z (machinery only).

## Next (reviewer decision, NOT authorized)

After governance approval, acquire ≥24 analyzable subjects under the pinned table (6–10 checkpoint is operations-only), then
run analysis A. **NOT authorized:** 6–10-only scientific pilot, `min_confirm_pairs` relaxation / gate tuning,
p/mean-T/feature/statistic changes, post-hoc balancing on observed Y, Lee2019 as validation, existing-dataset route as
primary, paper writing. Related: `../b9_stage1_prospective_pilot/` (B9.1A), `../b9_stage0_contract_design/` (B9.0),
`csc/b9/CONTRACT.md`.
