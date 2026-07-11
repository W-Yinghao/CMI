# STAR_00 Project Charter

## Identity

**Project:** STAR-EEG — Source-Task Anchored Representation Reorganization for EEG.

**Question:** Can a source-only task anchor move an H200 foundation representation from subject-identifiable but task-floor to task-transferable, without requiring subject erasure?

STAR is independent from S2P but depends on S2P's frozen scientific facts at commit `a9134eb5eb7f8486a5e1ee41831823dab39381ed`. STAR does not alter or supersede S2P Phase B or the S2P Phase B/Phase B-next control plane.

## Frozen motivation

The following are dependencies, not STAR results:

- H200 frozen FACED transfer remains at the random floor.
- H500 is the first sampled above-floor budget; H500/H1000/H2000 remain above floor.
- Pairwise subject separability L1 is about 0.979 at H200 and about 0.99 thereafter.
- Transfer emergence did not coincide with reduced subject separability.
- All eight frozen checkpoints passed the task gate.
- No checkpoint-level L5 measured-subject intervention exceeded its source-val-energy-matched random null after Holm correction.
- The budget curve is non-monotone and is not a scaling law.
- Released CBraMod is a descriptive frozen-reference band only; no equivalence, reproduction, or superiority claim is allowed.

## Hypotheses

- **H1:** H200's main gap is not excessive subject information, but task structure not organized for cross-subject access.
- **H2:** A small source-only task anchor on a fixed H200 checkpoint can produce frozen transfer earlier than optimizer-update- and batch-count-matched native SSL continuation. This is not a strict FLOP-matched claim.
- **H3:** A gain need not reduce subject L1. The stronger mechanism pattern is increased target transfer with high L1 and measured subject-subspace reliance no greater than the matched random null.
- **H4:** The true-label anchor must beat the fixed shuffled-label anchor. Otherwise a gain cannot be attributed to task semantics.

## Current authority

STAR_00A and STAR_00B are accepted infrastructure/design and real-path preflights. STAR_00C is authorized only for launch locking, persistent provenance, final-code bounded smoke, and the executable array-to-closure chain. After a STAR_00C machine PASS and a commit/artifact-bound approval lock, one blind six-cell STAR_01A array plus its `afterok` immutable closure is conditionally authorized. FACED target scoring, a scientific result table, a P1 claim, and manuscript work remain unauthorized.

## Success definition for STAR_00A

STAR_00A succeeds when the dependency is exact, protected projects remain unchanged, the required checkpoint inventory is auditable, split/compute/shuffle/firewall contracts are deterministic and fail closed, and the synthetic code path has finite losses and gradients. Artifact readiness and scientific approval are separate: even a fully passing preflight leaves `star01_approved = false`.
