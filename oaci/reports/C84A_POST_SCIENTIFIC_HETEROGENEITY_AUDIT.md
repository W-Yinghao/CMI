# C84A Post-Scientific External Heterogeneity Audit

## Final Audit Gate

```text
C84_POST_SCIENTIFIC_HETEROGENEITY_AND_TPAMI_THEORY_BRIDGE_AUDIT_COMPLETE_C85_PROTOCOL_REVIEW_REQUIRED
```

C84A is a read-only, post-outcome synthesis. It preserves the confirmatory C84S
gate `C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous` and frontier tag `C84-L4`. All new order statistics,
component distances and cross-cycle matrices are explicitly `POST_C84S_EXPLORATORY_DESCRIPTIVE`.

## Frozen Result Replay

- C84S lock, authorization, selection freeze, scientific result, and result
  manifest replay exactly.
- All 18 registered tables replay by SHA-256 and row count.
- Coverage remains 944 contexts and 18,432 method-context rows.
- Stage A/B/C lifecycle and protected counters replay without mismatch.
- Historical C84S scientific/final red teams and accepted regressions replay.

No EEG array, direct label view, target logit/source array, selector, Q0 builder,
Stage B/C callable, inference engine, model checkpoint, or oracle was opened.

## Audit Semantics

The full-panel matrix copies frozen max-T p-values and simultaneous Q2 bounds,
then replays each registered threshold as a signed distance. The level table
copies its frozen Q1/Q2 pass and p-values and computes only descriptive target
and panel/seed summaries from frozen result rows. A level-specific Q2
simultaneous upper bound was not frozen; it is marked
`NOT_FROZEN_AT_LEVEL_SCOPE` and is not reconstructed. Quantiles use linear
interpolation at `(n-1)p`. No permutation stream, max-T statistic, confidence
bound, selector score, or candidate choice is recomputed.

## Complete Zero-Label Gate Map

| Dataset | Method | Q1 | Q1 nearest failure | Q2 | Q2 nearest failure |
|---|---|---|---|---|---|
| Lee2019_MI | U5 NuclearNorm | FAIL | Q1_favorable | FAIL | Q2_target_count |
| Lee2019_MI | U7 ATC | FAIL | Q1_favorable | FAIL | Q2_target_count |
| Lee2019_MI | U11 MaNo | FAIL | Q1_favorable | FAIL | Q2_worst |
| Lee2019_MI | U13 COTT | FAIL | Q1_worst | PASS | NONE |
| Lee2019_MI | U14 SND | FAIL | Q1_favorable | FAIL | Q2_panel_seed |
| Lee2019_MI | U15 Agreement-on-the-Line | FAIL | Q1_favorable | FAIL | Q2_worst |
| Cho2017 | U5 NuclearNorm | FAIL | Q1_favorable | FAIL | Q2_target_count |
| Cho2017 | U7 ATC | FAIL | Q1_favorable | FAIL | Q2_target_count |
| Cho2017 | U11 MaNo | PASS | NONE | PASS | NONE |
| Cho2017 | U13 COTT | FAIL | Q1_worst | PASS | NONE |
| Cho2017 | U14 SND | FAIL | Q1_favorable | FAIL | Q2_target_count |
| Cho2017 | U15 Agreement-on-the-Line | FAIL | Q1_favorable | FAIL | Q2_target_count |
| PhysionetMI | U5 NuclearNorm | FAIL | Q1_favorable | FAIL | Q2_target_count |
| PhysionetMI | U7 ATC | FAIL | Q1_favorable | FAIL | Q2_target_count |
| PhysionetMI | U11 MaNo | FAIL | Q1_favorable | FAIL | Q2_target_count |
| PhysionetMI | U13 COTT | FAIL | Q1_worst | PASS | NONE |
| PhysionetMI | U14 SND | FAIL | Q1_favorable | FAIL | Q2_target_count |
| PhysionetMI | U15 Agreement-on-the-Line | FAIL | Q1_favorable | FAIL | Q2_target_count |

The nearest-failure label is a navigation aid, not a replacement gate. It uses
normalized distance only among failed registered components; all raw signed
distances and every pass flag remain in
`full_panel_gate_component_matrix.csv`. The level matrix preserves unavailable
components rather than inventing finite values.

## COTT: Positive Average, Non-Robust Tail

| Dataset | Mean | Median | Lower decile | Worst | Floor breaches | Frozen Q1 | Frozen Q2 |
|---|---:|---:|---:|---:|---:|---|---|
| Lee | 0.148038 | 0.161908 | 0.011322 | -0.107873 | 1 | FAIL | PASS |
| Cho | 0.181005 | 0.208972 | 0.014133 | -0.305171 | 1 | FAIL | PASS |
| Physionet | 0.096640 | 0.111489 | -0.107382 | -0.310447 | 9 | FAIL | PASS |

Lee has two adverse COTT targets, but its only registered floor breach is target 8 at
`-0.107873`, only 0.007873 below the -0.10
floor. Omitting target 8 is the sole Lee LOTO category change, C to A. Cho also
has two adverse targets; its only floor breach is target 3 at `-0.305171` and is
materially deeper, although Cho remains category A through U11/MaNo.
Physionet has 19 adverse targets, including nine floor-breaching targets
(`17|38|41|57|58|68|74|86|102`), with a minimum
of `-0.310447`. Its 76/76 LOTO categories
remain C. Thus Lee is a single-target near-boundary COTT failure, while the
Physionet tail is distributed; target composition is not the sole explanation
of C84-D.

COTT's mean direction is positive and its frozen Q2 decision passes in all
three cohorts. It nevertheless fails Q1 in every cohort. This is a recurrent
average-case/non-robust-tail pattern, not a Q1 success and not a universal
failure statement.

## MaNo: Decision Without Global Rank Fidelity

Cho U11/MaNo has mean regret `0.340079`,
Q1 mean `0.197703`, frozen Q1/Q2 PASS, and
worst target `-0.016765`. Its mean
Spearman is `0.000960` and top-1/top-5/top-10
are `0.0500` / `0.1562` /
`0.2437`.

All Cho MaNo selections are in the ERM regime, and its frozen selected utility
and regret exactly match B1 in all Cho contexts. This is a descriptive
selection-concentration fact. It does not identify why the policy works, nor
does it establish a dense near-optimal candidate region: the allowed compact
tables do not contain all 81 candidate utilities per context.

## Passive-Label Frontier

- Lee has no B*. B=8 and FULL satisfy the mean and frozen max-T components but
  fail the registered worst-target floor; lower budgets fail additional
  components. A lower mean Q0 FULL regret is therefore insufficient for the
  robust compound gate.
- Cho has B*=8 because B=8 and FULL directly qualify and satisfy larger-budget
  closure. Its frozen level frontiers differ: level 0 is FULL and level 1 is 4.
- Physionet has no B*. At B=8 and FULL the mean component is positive, but
  max-T, favorable-target and worst-target components fail.

The policy comparison is nonmonotone in observed regret under fixed,
non-nested registered policies. It is not a Blackwell comparison. The exact
decision-theory boundary is in `C84A_TPAMI_DECISION_THEORY_BRIDGE.md`.

### Complete Primary-Budget Component Replay

| Dataset | Budget | Mean effect | max-T p | Favorable | Worst (target) | Direct | Closure |
|---|---|---:|---:|---:|---:|---|---|
| Lee2019_MI | 1 | -0.007260 | 0.999954 | 9/22 | -0.248229 (8) | FAIL | FAIL |
| Lee2019_MI | 2 | 0.034045 | 0.924607 | 12/22 | -0.225381 (8) | FAIL | FAIL |
| Lee2019_MI | 4 | 0.067710 | 0.378015 | 19/22 | -0.195791 (8) | FAIL | FAIL |
| Lee2019_MI | 8 | 0.097640 | 0.0484154 | 20/22 | -0.165708 (8) | FAIL | FAIL |
| Lee2019_MI | FULL | 0.138665 | 0.0101622 | 18/22 | -0.151660 (8) | FAIL | FAIL |
| Cho2017 | 1 | 0.030246 | 0.965256 | 14/20 | -0.187817 (3) | FAIL | FAIL |
| Cho2017 | 2 | 0.101076 | 0.0503838 | 17/20 | -0.161868 (3) | FAIL | FAIL |
| Cho2017 | 4 | 0.162283 | 0.000473015 | 18/20 | -0.115694 (3) | FAIL | FAIL |
| Cho2017 | 8 | 0.216546 | 3.05171e-05 | 19/20 | -0.062050 (3) | PASS | PASS |
| Cho2017 | FULL | 0.328576 | 1.52586e-05 | 20/20 | 0.056816 (3) | PASS | PASS |
| PhysionetMI | 1 | 0.031608 | 0.974274 | 50/76 | -0.171579 (17) | FAIL | FAIL |
| PhysionetMI | 2 | 0.044408 | 0.7952 | 51/76 | -0.156090 (38) | FAIL | FAIL |
| PhysionetMI | 4 | 0.056140 | 0.454262 | 53/76 | -0.168590 (61) | FAIL | FAIL |
| PhysionetMI | 8 | 0.067789 | 0.192944 | 52/76 | -0.191414 (61) | FAIL | FAIL |
| PhysionetMI | FULL | 0.069567 | 0.214657 | 49/76 | -0.270568 (57) | FAIL | FAIL |

The favorable counts, worst rows, and panel/seed cell means are exact
descriptive reductions of frozen C84S result rows. The displayed max-T p-values,
direct decisions, closure decisions, and B* values are copied from the frozen
frontier table; none is re-estimated.

## Heterogeneity Synthesis

C82 and C84 together show heterogeneity across training seed, dataset,
source-support level, target composition and decision objective. No
cross-dataset p-value is introduced. Method identity is retained row by row in
`c82_c84_method_transport_matrix.csv`.

The transport matrix contains 45 identity-preserving rows: nine methods across
BNCI2014_001 seeds 3/4 and the three C84 cohorts. C80 supplies the frozen Q0
rows for the BNCI field; C82 supplies the strict-source and zero-label rows;
C84S supplies the external-cohort rows. Missing or semantically incomparable
measurements are labelled as such. The matrix performs no pooling and creates
no cross-study p-value.

LOTO category preservation is 21/22,
20/20, and
76/76. Lee's level-specific
COTT disagreement and the A/C/C dataset categories already suffice for C84-D,
so the single Lee LOTO category change does not carry the full heterogeneity
interpretation.

## Theory And Next Evidence

The highest-priority gaps are a formal information-experiment comparison, a
robust target-risk formulation, and a prospective active-versus-passive label
policy. Each requires assumptions and untouched populations not supplied by
C84A. The decision matrix is advisory only: C85, active acquisition, new data,
new model zoos, and manuscript work remain unauthorized.

## Verification

The read-only generator passed 75/75 internal checks and its focused test file
passed 14/14. Accepted Slurm regressions at implementation commit `40764599`
were:

```text
focused:  256 passed
C65:      867 passed, 1 skipped, 3 deselected
C23:    1,278 passed, 1 skipped, 3 deselected
full:   2,202 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. The initial wrong-environment focused
attempt is preserved in `C84A_REGRESSION_VERIFICATION.md` and is not counted as
accepted. `squeue` showed zero active C84/C85 jobs after completion; `sacct`
was not used.

## Claim Boundary

Supported: COTT has cross-cohort positive-average value with registered tail
failure; MaNo has Cho-specific decision value without global ranking fidelity;
passive-label actionability and source-support effects are nonuniform under
fixed policies.

Not supported: universal method success/failure, universal external validity,
an information-theoretic ordering of labels and unlabeled outputs, a proved
Blackwell/minimax theorem, a general benefit of support deletion, or any C85 or
manuscript authorization.
