# C78S — Full Seed-3 Multi-Regime Scientific Analysis

## Final gate

```text
SEED3_MIXED_RESULTS_C79_PROTOCOL_REVIEW_REQUIRED
```

Primary target field: eight prospectively generated seed-3 targets, 1,296 units.
Target 4 is excluded from every primary estimand, null pool, and multiplicity family.
Seed 3 remains exploratory replication; it is not an independent target-population or seed confirmation.

## Registered H1–H6 verdicts

| Hypothesis | Active | Effect | Raw p | Holm p | Boundary |
|---|---:|---:|---:|---:|---|
| H1 | 0 | 0.756456 | 0.011673 | 0.070039 | actionability_nonqualification_not_null_acceptance |
| H2 | 0 | -8.717406 | 0.862000 | 1.000000 | not_applicable |
| H3 | 0 | 0.210137 | 0.092000 | 0.368000 | transport_and_actionability_gates_not_p_greater_0.05 |
| H4 | 1 | -0.096288 | 1.000000 | 1.000000 | registered_candidate_nonqualification_only |
| H5 | 1 | 0.010450 | 1.000000 | 1.000000 | registered_candidate_nonqualification_only |
| H6 | 0 | 0.415635 | 0.011673 | 0.070039 | not_applicable |

## Measurement and control

Construction/evaluation trajectory reliability is `0.756456`
(target-cluster 95% CI `0.681795` to
`0.833763`). The registered construction score's
top-5 improvement over the source baseline is `0.500000`;
standardized regret reduction is `0.686781`.
These quantities are reported separately because association/reliability is not checkpoint control.

## Representation information classes

Strict-source F2 incremental R2 is `-0.096288`;
target-unlabeled F4-geometry incremental R2 is
`0.010450`. Registered qualification requires
material R2, corrected-null passage, leave-target and leave-regime transport, positive direction in
at least six targets, and material top-k or regret improvement. The report therefore makes only the
registered candidate/nonqualification calls shown in H4 and H5; it does not claim universal
impossibility for all source or target-unlabeled functions.

## Geometry and transport

The effective-multiplicity/top-gap model improves held-target top-1-miss deviance by
`-8.717406` with blocked permutation p
`0.862000`. This is endpoint-derived diagnostic geometry, not a selector.
The strongest registered local nonlinear association is
`0.210137`; its prediction and actionability gates are reported
separately and cannot be replaced by an association p-value.

## Information and provenance boundaries

- Construction and evaluation trial IDs are physically disjoint and cover all 576 target trials.
- Trial IDs and row order are used only for joining, splitting, and dependence clustering.
- The same-label oracle descriptor was not presented to the primary runner and was never opened.
- No training, forward pass, re-inference, GPU, seed 4, C79 execution, BNCI2014_004, selector,
  checkpoint recommendation, or manuscript drafting occurred.
- C79 is protocol-ready only. It remains unauthorized.

## Taxonomy

```text
C78S-H4_no_registered_strict_source_representation_escape_hatch
C78S-H5_no_registered_target_unlabeled_representation_actionable_control
C78S-S1_complete_1296_unit_primary_field_consumed
C78S-S2_target4_mechanically_excluded
C78S-S3_same_label_oracle_not_accessed
C78S-S4_seed3_exploratory_not_confirmation
C78S-S5_seed4_untouched
C78S-S6_C79_protocol_ready_execution_not_authorized
```

## Red-team status

The independent result red team must pass before this report is presented as final.
