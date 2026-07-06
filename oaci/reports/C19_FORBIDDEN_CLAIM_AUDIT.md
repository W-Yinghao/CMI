# C19 — forbidden-claim audit

This probe is DIAGNOSTIC-ONLY and emits NO selector. The report text is checked against the forbidden-claim list; the no-selector gate is asserted.

- FORBIDDEN: deployable target-free selector
- FORBIDDEN: deployable selector
- FORBIDDEN: validates a selector
- FORBIDDEN: oaci is rescued
- FORBIDDEN: target oracle is deployable
- FORBIDDEN: support mismatch caused the original oaci failure
- FORBIDDEN: all dg fails
- FORBIDDEN: eeg transfer is impossible
- FORBIDDEN: support-aware invariance is useless
- FORBIDDEN: we built a selector
- FORBIDDEN: we found a selector
- FORBIDDEN: competence detector is validated

- no_selector_artifact gate: True
- static excluded from primary: True
- fragile endpoints excluded from primary: True
