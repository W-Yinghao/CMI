# B9.3 / B9.GOV governance status

Machine-checkable status only. No personal/sensitive information (no PII, no signed consent, no reviewer identities, no
institution-private correspondence). Does not grant approval and is not legal/IRB advice. The **authoritative, machine-readable
tracker is `b9_gov_status.json`** (8-state model + allowed transitions + pinned frozen-protocol hashes + repo-hygiene rules).

```
status: not_started
```

Allowed states: `not_started` | `drafting_external_submission` | `submitted` | `revisions_requested` | `approved` |
`not_approved` | `withdrawn` | `expired`.

**No real human-EEG data has been collected. Real acquisition MUST NOT begin until `status: approved`** through the
institution's ethics/governance, informed-consent, data-handling, and lab-safety processes. The assistant's authorization
covers protocol preparation + analysis plan + simulated non-human DAQ readiness + governance-status tracking ONLY.

**Protocol-change trigger:** if the governance review requests modifications affecting assignment, trial/microblock structure,
exclusions, or recorded data fields, those changes MUST trigger a protocol diff vs the pinned hashes in `b9_gov_status.json`
AND a design red-team, BEFORE any acquisition. **After approval**, the next authorized stage is B9.4 (6–10 subject
operations-only checkpoint, scientific endpoints structurally blinded); Analysis A remains `PENDING_ACQUISITION`.
