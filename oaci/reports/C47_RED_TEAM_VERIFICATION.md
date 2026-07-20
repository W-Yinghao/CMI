# C47 - Red-Team Verification

C47 red-team checks were run after artifact generation and before commit.

- group_conditioned_baseline_audit: pass - Each grouped top-k row uses the same group's exact random baseline.
- conditioning_problem_class_boundary: pass - Target/trajectory grouping is treated as diagnostic conditioning, not target-free action.
- source_smoothing_not_tuned: pass - Smoothing uses inherited C45 q10 source-neighborhood radius only.
- hindsight_and_oracle_disclosure: pass - C43 best scalarization and target utility ceiling are explicitly diagnostic rows.
- reliability_gate_audit: pass - Best conditioned strict-source top1 remains below the reliability gate.

Verdict: C47 is diagnostic-only. The target oracle ceiling and C43 hindsight scalarization remain disclosed as ceilings, not deployed actions.
