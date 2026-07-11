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
| H1 | 0 | 0.770863 | 0.011673 | 0.058366 | actionability_nonqualification_not_null_acceptance |
| H2 | 0 | -9.505921 | 0.896000 | 1.000000 | not_applicable |
| H3 | 1 | 0.242656 | 0.002000 | 0.012000 | transport_and_actionability_gates_not_p_greater_0.05 |
| H4 | 1 | -0.073086 | 1.000000 | 1.000000 | registered_candidate_nonqualification_only |
| H5 | 1 | 0.005176 | 1.000000 | 1.000000 | registered_candidate_nonqualification_only |
| H6 | 0 | 0.404329 | 0.019455 | 0.077821 | not_applicable |

## Measurement and control

Construction/evaluation trajectory reliability is `0.770863`
(target-cluster 95% CI `0.699990` to
`0.841695`). The registered construction score's
top-5 improvement over the source baseline is `0.687500`;
standardized regret reduction is `0.696682`.
These quantities are reported separately because association/reliability is not checkpoint control.

### Registered counter-result

H1 did **not** replicate the prior measurement-control separation. The reason is
not weak measurement: construction/evaluation reliability is high. The reason is
that the split-label construction information became materially useful inside
this registered diagnostic information class:

```text
true best recovered in construction top-1:   0.1250  (random 0.0123)
true best recovered in construction top-5:   0.6875  (random 0.0617)
true best recovered in construction top-10:  0.7500  (random 0.1235)
construction standardized regret:            0.0828
random expected standardized regret:         0.4820
```

The crossed target-by-trial bootstrap gives reliability `0.6925`
(`95% CI 0.6060–0.7668`), top-5 hit `0.4732` (`0.1250–0.8125`), and standardized
regret `0.1356` (`0.0578–0.2240`). H1 also narrowly misses family-wise significance
(`Holm p=0.0584`), but its primary failure is the pre-locked material-actionability
gate. This is a target-construction-label diagnostic counter-result. It is not a
source-only selector, deployability result, or OACI rescue.

H6 is descriptively strong (`incremental R2=0.4043`, raw `p=0.0195`) but remains
inactive after H1-H6 Holm correction (`p=0.0778`). It must not be promoted to a
confirmed strongest-positive-control claim.

## Representation information classes

Strict-source F2 incremental R2 is `-0.073086`;
target-unlabeled F4-geometry incremental R2 is
`0.005176`. Registered qualification requires
material R2, corrected-null passage, leave-target and leave-regime transport, positive direction in
at least six targets, and material top-k or regret improvement. The report therefore makes only the
registered candidate/nonqualification calls shown in H4 and H5; it does not claim universal
impossibility for all source or target-unlabeled functions.

## Geometry and transport

The effective-multiplicity/top-gap model improves held-target top-1-miss deviance by
`-9.505921` with blocked permutation p
`0.896000`. This is endpoint-derived diagnostic geometry, not a selector.
The strongest registered local nonlinear association is
`0.242656`; its prediction and actionability gates are reported
separately and cannot be replaced by an association p-value.

The local signal is specifically target-unlabeled geometry: Laplacian-HSIC is
`0.2427` within target×level×trajectory, positive in all 32 eligible trajectory
cells, and passes all six registered blocked controls (`worst max-stat p=0.002`).
It still harms fixed-kernel held-target prediction (`incremental R2=-0.2129`) and
held-regime prediction (`-0.0858`), so H3 remains association-only and
nontransportable. The strict-source fixed association does not survive the same
controls (`worst p=1.0`).

H2 is inactive: adding effective multiplicity and top-gap terms worsens held-target
top-1-miss deviance by `9.5059` relative to raw candidate count
(`permutation p=0.896`). This does not erase older within-universe near-tie findings;
it says the registered C78S cross-target failure model did not transport.

## Information and provenance boundaries

- Construction and evaluation trial IDs are physically disjoint and cover all 576 target trials.
- Trial IDs and row order are used only for joining, splitting, and dependence clustering.
- The same-label oracle descriptor was not presented to the primary runner and was never opened.
- No training, forward pass, re-inference, GPU, seed 4, C79 execution, BNCI2014_004, selector,
  checkpoint recommendation, or manuscript drafting occurred.
- C79 is protocol-ready only. It remains unauthorized.

## Taxonomy

```text
C78S-H3_local_nonlinear_association_nontransportable_nonactionable
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

Independent result red team: PASS (60/60, zero blockers).

## Regression

```text
focused:    43 passed
C65-C78S:  256 passed, 1 conditional skip
C23-C78S:  663 passed, 1 conditional skip
full OACI: 1591 passed, 1 conditional skip
```

All four jobs used `cpu-high`, 48 CPUs, and produced empty stderr logs.
