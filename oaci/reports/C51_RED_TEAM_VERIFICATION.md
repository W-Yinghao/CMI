# C51 - Red-Team Verification

C51 red-team checks were run after artifact generation and before commit.

- locked_witness_replayed: pass - C51 replays the C50 q20/min_n=1 witness and does not search for a new ceiling.
- self_neighbor_excluded: pass - Within-target distance matrices keep query rows excluded from their own neighborhoods.
- target_labels_quarantined: pass - Label shuffles and score transforms are diagnostic controls, not deployable rules.
- null_calibration_complete: pass - N0-N4 null summaries are emitted for all required statistics.
- support_grid_complete: pass - Support ablation covers q10/q20/q30/q40 and min_n 1/2/3/5.
- source_score_attribution_complete: pass - Available source score families are audited with sign, monotone, and grouped diagnostic controls.
- no_selection_artifact: pass - Tables omit selection identifiers and recommendation fields.
- no_deployable_claim: pass - C51 is reported as failure attribution and source-describability diagnostics only.

Verdict: C51 is a diagnostic failure-attribution audit over the locked C50 witness.
