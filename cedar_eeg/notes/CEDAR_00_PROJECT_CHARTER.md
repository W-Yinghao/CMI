# CEDAR-EEG Project Charter

CEDAR is a method-validation project. It does not write manuscript text. It
produces code, audits, protocols, and evidence records that a PM can review from
GitHub.

## Objective

Locate subject/session/site fingerprints in EEG decoder representations and
test whether source-validated structural deletion can remove extractable
conditional domain information without damaging task performance.

## Starting Phase

P0 frozen latent mask only. No backbone training, no new CMI regularizer, no
deployment router, no target-label threshold selection.

## Dependencies To Reuse As Discipline

- leakage audit: conditional domain probes and permutation nulls
- R3 reliance: deletion must not increase functional task reliance
- TOS: task-protected deletion and identity fallback
- CSC: abstain when not identifiable
- ACAR: no-label deployment API discipline
- CITA/TTA: TTA-Control is a downstream comparator only
- CutClean: block heads and structured pruning inspire P1, not P0

## Explicit Exclusions

`oaci`, `h2cmi`, and FSR are not CEDAR dependencies.
