# C44 Red-Team Verification

## Scope

C44 was checked as a read-only diagnostic audit over inherited C43 source objectives and C43/C42/C41/C35/C32R artifacts. It did not train, fine-tune, re-infer, add objectives, run feature selection, emit selected-checkpoint artifacts, or change the frozen C19 config hash `664007686afb520f`.

## Red-Team Findings

- **F1 containment caveat held.** The audit separates target-good-on-front from source-side identifiability: the observed source-front fraction is 0.972, so front membership is nearly saturated.
- **High-dimensional null check passed.** Observed source-front fraction 0.972 is close to the frozen Gaussian same-dimension null 0.981 and objective-shuffled null 0.981, supporting PF1/PF9 rather than a new localization rule.
- **Front membership is not discriminative.** P(joint-good | front) is 0.431 against trajectory baseline 0.430, with P(joint-good | not-front) 0.529; front co-occupancy of joint-good and joint-bad trajectories is 0.889.
- **Objective conflict, not simple redundancy, explains the broad front.** Effective rank is 5.696 over 10 inherited objectives, negative objective-pair fraction is 0.444, and leakage/rank family mean Spearman is -0.071.
- **PF6 is diagnostic-only.** Rank-only reduction narrows the front to 0.044 and has weak signal (joint-good enrichment 1.154, depth AUC 0.590), but it is not a selector and does not repair the full source-front degeneracy.
- **Dominance depth is non-actionable.** Full-objective dominance-layer AUC against target utility is 0.499, so PF8 remains inactive and PF7 is the conservative call.
- **No method artifact emitted.** Tables avoid checkpoint/model hashes and selected-candidate identifiers; `no_selector_artifact_gate.csv` records all gates as passed.

## Verification

- `py_compile`: passed for `oaci/source_frontier_geometry/*.py` and `oaci/tests/test_c44_source_frontier_geometry.py`.
- Slurm focused job `890137` on `cpu-high`: `10 passed in 0.16s`.
- Slurm regression job `890138` on `cpu-high`: `202 passed in 32.24s` for C23-C44.

## Conservative Taxonomy

Accepted:

```text
PF1_source_pareto_front_degenerate
PF2_front_membership_non_discriminative
PF3_objective_conflict_inflates_front
PF4_objective_redundancy_not_the_issue
PF5_family_reduced_frontier_narrows_but_loses_target_coverage
PF6_family_reduced_frontier_has_diagnostic_signal
PF7_dominance_depth_not_target_informative
PF9_source_objective_geometry_non_identifiable
```

Not active:

```text
PF8_dominance_depth_weakly_informative_but_non_actionable
PF10_inconclusive_due_to_objective_availability
```
