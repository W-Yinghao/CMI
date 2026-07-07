# B9.3 data management plan — DRAFT template (development-only)

> **Gate:** real collection is gated on institutional approval BEFORE any recording; governance `status: not_started` (`b9_stage3_governance_status.md`).

> DRAFT to be aligned to the institution's data-governance / privacy policy. Not approval, not legal/IRB advice.

- **What is recorded:** raw EEG + the frozen event/marker log + the companion trial log (subject_slot, microblock, trial_id,
  C, Y_design, timing, artifact_flag). `Y_design` is the pre-assignment cue, never an observed post-hoc label.
- **Identifiers / de-identification:** participants are keyed by an opaque `subject_slot`; any enrollment key linking slot to
  identity is held separately under institutional access control and is **never** committed to this repository.
- **Storage + access:** raw data stored on institutionally-approved secure storage with role-based access; the repository
  holds only the pre-registration, code, and non-PII synthetic/dry-run artifacts.
- **Retention / deletion:** retention period + secure deletion per institutional policy; withdrawal-triggered deletion clause
  stated to the local template.
- **Sharing constraints:** any data sharing follows the institution's approvals + consent scope; no raw human EEG is placed in
  this repository.
- **Integrity:** the pinned assignment table (`.sha256`) and event-marker map are hash-frozen; executed data are audited by
  trial_id join (adherence) before any analysis.
