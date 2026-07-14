# Review Completion Current Status

- canonical repaired-W1 scientific head: `cfb43d429263c3fbb69a35c086214fdca7d25301`
- canonical FP-GEM result head: `b5fb51581b634a194f395aa151babf4f5817ccae`
- experimental phase: closed
- main method: Fixed-Prior Geometry EM (FP-GEM)
- legacy W1 split: quarantined
- repaired H2CMI W1: confirmatory
- official SPDIM repaired-split three-seed baseline: complete
- official SPDIM blocker: resolved by P9
- cross-pipeline interpretation: same-split full-pipeline only
- adapter-only H2CMI versus SPDIM comparison: not supported
- equivalence/noninferiority: not supported
- additional GPU required: `false`
- additional experiment required: `false`
- active stale errors: `0`

## Canonical Entry Points

- Scientific freeze: `FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md/json`
- All writer-facing numbers: `MANUSCRIPT_NUMBERS_READY.md`
- FP-GEM writer-facing story: `h2cmi/results/fp_gem_main/FINAL_FP_GEM_STORY_FREEZE.md/json`
- FP-GEM same-backbone table: `h2cmi/results/fp_gem_main/fp_gem_final_head_to_head.md/csv`
- FP-GEM evidence hierarchy: `h2cmi/results/fp_gem_main/FP_GEM_EVIDENCE_HIERARCHY.md/json`
- P12 canonical same-backbone source: `h2cmi/results/fp_gem_main/fp_gem_results.csv`
- P13 canonical appendix boundary: `h2cmi/results/fp_gem_prevalence/P13_MANUSCRIPT_BOUNDARY.md/json`
- Artifact status map: `CANONICAL_EVIDENCE_INDEX.md/json`
- Stale-claim gate: `STALE_CLAIM_AUDIT.md/json`

## FP-GEM Claim Boundary

- P12 supports improvement over source-only, not improvement over RCT or SPDIM.
- P12 does not support FP-GEM superiority over Joint-GEM on natural transfer.
- P13 does not support lower sensitivity than Joint-GEM or SPDIM.
- P13 supports only lower sensitivity than RCT on the frozen endpoint, with lower endpoint performance than the strongest baselines.
- Universal prevalence robustness, state-of-the-art, equivalence, and noninferiority claims are prohibited.

## Active Blockers

1. Orthogonal-score estimator/evaluation.
2. Montage-layout or cross-montage remapping stress.

No experiment is required or authorized by this status pointer.
