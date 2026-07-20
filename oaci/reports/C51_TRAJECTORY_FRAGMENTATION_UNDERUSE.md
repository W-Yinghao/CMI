# C51 - Trajectory Fragmentation / Source-Describability Audit (frozen C19 `664007686afb520f`)

## Decision

`C51-E_target_trajectory_gauge_residual`

## Locked Witness Replay

- condition/source/eps/min_n: `within_target / all_source_objectives / q20 / 1`
- epsilon: **3.253**
- C50 hit / coverage / enrichment: **1.000 / 1.000 / 2.360**
- trajectory min hit / coverage: **0.000 / 1.000**

## Attribution

- support material: **False**.
- null-like trajectory fragmentation: **False**.
- stronger than N2/N3 nulls: **False**.
- best raw underuse gap: **0.401**.
- best target/trajectory diagnostic control gaps: **0.244 / -0.004**.
- N2/N3 fail-fraction percentiles: **0.000 / 0.000**.
- N4 source-geometry permutation enrichment mean: **0.834**.

## Bottom Line

C51 attributes C50 trajectory fragmentation as a diagnostic source-describability boundary. The trajectory fail fraction itself is not worse than the N2/N3 nulls, but source-geometry permutation collapses the enrichment and existing source scores leave a large underuse gap that only trajectory-conditioned diagnostic controls close. The locked witness remains real, but the audit does not create a source-only selection rule.

## Red-Team Checks

- locked_witness_replayed: PASS - C51 replays the C50 q20/min_n=1 witness and does not search for a new ceiling.
- self_neighbor_excluded: PASS - Within-target distance matrices keep query rows excluded from their own neighborhoods.
- target_labels_quarantined: PASS - Label shuffles and score transforms are diagnostic controls, not deployable rules.
- null_calibration_complete: PASS - N0-N4 null summaries are emitted for all required statistics.
- support_grid_complete: PASS - Support ablation covers q10/q20/q30/q40 and min_n 1/2/3/5.
- source_score_attribution_complete: PASS - Available source score families are audited with sign, monotone, and grouped diagnostic controls.
- no_selection_artifact: PASS - Tables omit selection identifiers and recommendation fields.
- no_deployable_claim: PASS - C51 is reported as failure attribution and source-describability diagnostics only.
