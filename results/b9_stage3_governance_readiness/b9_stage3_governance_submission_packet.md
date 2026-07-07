# B9.3 governance submission packet — DRAFT template (development-only)

> **This is a study-design DRAFT to be adapted into the institution's own IRB / ethics / governance template.** It is **not**
> the submission itself, **not** approval, and **not** legal/IRB advice. If the institution has a fixed template, fill that
> template and treat this as source content only. **No real human-EEG data may be collected until the local governance
> process returns `approved`** (`b9_stage3_governance_status.md`, mirroring B9.2, remains `not_started`).

## 1. Study purpose

A **prospective randomized-audit contract feasibility and null-control audit** for EEG: verify that a pre-recording,
hash-pinned randomized assignment of an audit condition `C` and a pre-assigned cue class `Y_design`, executed and audited
against the pinned schedule, yields an exact-randomization null that runs and fails-closed on a real ≥24-analyzable cohort.

**What it is NOT:** clinical diagnosis; treatment or intervention; deployment/validation of a classifier; a biological or
cognitive-neuroscience concept claim. No participant receives any diagnostic or clinical output.

## 2. Participants

- **Inclusion / exclusion:** healthy adult volunteers able to give informed consent; standard EEG exclusions (e.g. scalp
  conditions precluding electrode placement) — to be finalized to the institution's standard. See `b9_stage3_recruitment_screen.md`.
- **Withdrawal:** participants may withdraw at any time without penalty; their data are handled per the withdrawal clause in
  `b9_stage3_data_management_plan.md`.
- **Compensation:** per institutional policy (state the amount/none in the local template).
- **Expected time burden:** one session; ~120 trials × 10 microblocks with breaks (estimate ≈ 45–75 min including setup);
  finalize to the local template.

## 3. EEG procedure

- **Equipment:** standard research EEG (montage/amplifier per lab), impedance + safety checks in `b9_stage3_lab_safety_checklist.md`.
- **Task / cue structure:** on each trial the participant is presented a **pre-assigned cue** (`Y_design`, the intended class);
  the audit condition `C` is a pre-randomized experimental manipulation. Both are fixed by the assignment table before recording.
- **Microblock structure:** trials are grouped into microblocks; within each microblock the four `(C, Y_design)` cells are
  equal (quadruplet × R=3), in the randomized order fixed by the table. Breaks between microblocks.
- **Trial count:** 30 enrollment slots × 120 trials = 3600 planned (target ≥24 analyzable; hard minimum ≥20).
- **Artifact handling:** predeclared automated rejection (see `b9_stage2_quality_criteria.json`); rejected trials are
  attrition, never post-hoc rebalancing.

## 4. Randomization (the crux)

- The assignment table is **generated before recording** and **hash-pinned** (`b9_stage2_assignment_table.csv` + `.sha256`;
  manifest attests `generated_before_recording=True`, `Y_design_pre_assignment=True`).
- Within each `subject × microblock`, `C × Y_design` is **exactly balanced**; order is randomized from the table.
- **`Y_design` is a pre-assignment cue / stimulus class — never an observed or generated post-hoc label.**
- Event markers are frozen (`b9_stage3_event_marker_map.json`); executed trials are audited against the pinned table by
  **trial_id join** (full-tuple adherence); any deviation → `CONTRACT_INVALID_OR_OUT_OF_ESTIMAND`.

## 5. Data handling

Detailed in `b9_stage3_data_management_plan.md`: raw EEG storage, identifiers / de-identification, access control, retention
/ deletion, and sharing constraints — to be aligned to the institution's data-governance policy.

## 6. Analysis (pre-registered; no adaptive changes)

- Pre-analysis protocol frozen (`b9_stage2_preanalysis_protocol.json`): contract-valid null audit (A), forced-violation
  validator tests (B), optional semi-synthetic positive diagnostic (C).
- **No adaptive gate changes. `min_confirm_pairs = 20` is unchanged.** The stop rule (≥24 analyzable OR 30 slots) and
  "analyzable" are fixed by predeclared quality/support, never by p-values/effect sizes.
- An early **6–10 subject operations-only checkpoint** inspects **only logistics** (adherence, attrition, artifact rate,
  support, sampler feasibility, contract-invalid reasons) and is **structurally blinded** to `B9_ALERT` / p-values /
  `observed_T` / `T_z` / condition-specific trends (`operations_checkpoint.checkpoint_report`).

## 7. Readiness evidence (non-human)

A **simulated, non-human DAQ dry-run** (`b9_stage3_daq_dryrun_rows.jsonl`, `b9_stage3_acquisition_app_checks.json`) verifies
the acquisition plumbing before any human recording: read-only assignment player, frozen unambiguous markers, trial_id-join
adherence, blinded checkpoint, and forced-corruption refusals. **No real EEG, no scientific endpoints.**

## 8. Governance gate

Real acquisition **must not begin** until the institution's ethics/governance, informed-consent, data-handling, and lab-safety
processes return `approved`. This technical package does not substitute for, and does not constitute, that approval.
