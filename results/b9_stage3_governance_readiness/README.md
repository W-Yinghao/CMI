# B9.3 governance submission + acquisition-readiness lock (diagnostic-only, NO real data / NO recording)

Reviewer-authorized 2026-07-07 after B9.2. The project has reached the point where **the only remaining information gain is a
governed, prospective randomized-audit acquisition** — so B9.3 *locks readiness* rather than producing science. It is a
**governance submission packet + a simulated, non-human DAQ dry-run.** **No real human-EEG data has been collected. No
scientific/biological/power claim is made. This is NOT validation.**

> **Hard gate.** Real human-EEG collection must not begin until the institution's ethics/governance, informed-consent,
> data-handling, and lab-safety processes return `approved` (`b9_stage3_governance_status.md` — `status: not_started`). The
> governance documents here are **DRAFT templates to adapt to the institution's own template** — **not** the submission,
> **not** approval, **not** legal/IRB advice. The institution's process controls collection.

## Contents

- **Governance drafts** (adapt to local template): `b9_stage3_governance_submission_packet.md`, `b9_stage3_consent_outline.md`,
  `b9_stage3_recruitment_screen.md`, `b9_stage3_data_management_plan.md`, `b9_stage3_lab_safety_checklist.md`,
  `b9_stage3_governance_status.md` (`not_started`).
- **DAQ readiness:** `b9_stage3_event_marker_map.json` (frozen cue codes 11/12/21/22 for `C × Y_design`; trial_id in the
  companion log), `b9_stage3_acquisition_app_checks.json`, `b9_stage3_daq_dryrun_rows.jsonl`.
- `b9_stage3_redteam_checks.json` — design red-team (`whlwgi1xo`, both lenses MINOR_ISSUE, no blockers).

## Simulated non-human DAQ dry-run (`DAQ_READY=True`)

Verifies the acquisition *plumbing* on mock event logs (**no real EEG, no Z, no statistical null, no alert**):
- **Read-only assignment player** — reads the pinned B9.2 30-slot table in row order and does **not** regenerate it
  (verified by both the composition `table_hash` and a **strict row-for-row / trial_id-sequence** attestation).
- **Frozen, unambiguous markers** — every trigger cue code equals `CUE_CODE[(C, Y_design)]`; `trial_id` is recoverable from
  the companion log (3600/3600 unique).
- **trial_id-join adherence** — the executed log is audited against the pinned table by trial_id join (full-tuple), `VALID`
  on a faithful recording.
- **Blinded operations checkpoint** — logistics only; never computes a p-value or alert.
- **All 7 forced corruptions refuse** — missing table, flipped C, shuffled `Y_design`, trial_id mismatch, microblock
  imbalance, condition-lock, marker-code mismatch → `CONTRACT_INVALID_OR_OUT_OF_ESTIMAND` (or marker-integrity), **before any
  p-value** (the DAQ path never calls the exact null).

The frozen B9.0/B9.2 machinery is reused **unchanged** (empty git diff); `min_confirm_pairs = 20` and `ALPHA = 0.025` are
unchanged. Two hashes appear and are distinct legitimate objects: the CSV **file** sha256 (`cc9b5780…`) and the manifest
**table_hash** (`e3675b16891e9b24`, the composition hash used by the validator).

## The gated sequence (reviewer-defined; only step 1 is done here)

1. **B9.3 governance/readiness packet** ✓ (this package).
2. Governance **approval** (institution).
3. **6–10 subject operations-only checkpoint** (logistics only, blinded).
4. Logistics red-team.
5. Explicit **go/no-go** for the full ≥24-analyzable acquisition.
6. **Analysis A** on real acquired data (**PENDING_ACQUISITION**).

**No shortcut from synthetic readiness to Analysis A.** Not authorized: real recording before approval, full acquisition
before the checkpoint review, `min_confirm_pairs` relaxation / gate tuning, statistic/feature changes, post-hoc balancing on
observed Y, Lee2019 as validation, existing-dataset route as primary, paper writing. Related:
`../b9_stage2_prospective_acquisition/` (B9.2), `csc/b9/CONTRACT.md`.
