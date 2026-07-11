# STAR_00 Acceptance Criteria

## STAR_00A required pass conditions

- Remote S2P dependency ref equals `a9134eb5eb7f8486a5e1ee41831823dab39381ed` exactly.
- STAR branch merge-base is that commit and branch name is `project/star-task-anchor`.
- No S2P scientific, H2CMI, OACI, or other non-STAR path is changed.
- Ten checkpoint/reference objects are inventoried with path/config, existence, repeated SHA, bytes, stability, strict reload, source commit, route/channel hashes, completion provenance, and role flags.
- Only H200_s0/s1 are valid starts; later budgets/released/random are reference-only.
- Exact FACED 1–80/81–100/101–123 firewall passes.
- Training function signatures contain source labels only and no target-label argument.
- B/C/D have equal optimizer steps and batch counts; B replaces every anchor slot with SSL; C/D schedules are identical and deterministic.
- Shuffled labels are fixed, source_train-only, within-subject, deterministic, and class-histogram preserving.
- Active method registry/import path contains only the approved native SSL/source-task scaffold.
- One toy SSL step and one toy source-task step produce finite loss and gradients deterministically without real EEG.
- `compileall`, all STAR tests, and diff whitespace checks pass.
- All ten small JSON artifacts and the readout are deterministic and internally hashed.

If checkpoint checks fail but the scaffold/firewalls pass, infrastructure may pass internally, but the externally reported status must be `STAR_01_BLOCKED_ARTIFACT_SUPPLY`. H200 may not be retrained to repair supply.

## Non-acceptance conditions

STAR_00A fails on a dependency mismatch, protected-path mutation, split ambiguity, target-data access surface, compute mismatch, mutable/non-deterministic shuffle, forbidden active method, non-finite synthetic smoke, or non-deterministic artifact generation.

## Approval separation

A STAR_00A PASS is not scientific evidence and is not STAR_01 approval. `star01_approved` must remain false. PM review is required before any real training, target scoring, SLURM action, or scientific table.
