# OACI / EEG-DG Project Memory Through C84A

## Current State

```text
C84S primary gate:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

C84S label-frontier tag:
  C84-L4

C84A audit gate:
  C84_POST_SCIENTIFIC_HETEROGENEITY_AND_TPAMI_THEORY_BRIDGE_AUDIT_COMPLETE_C85_PROTOCOL_REVIEW_REQUIRED
```

C84A is a read-only post-outcome audit. It is not independent confirmation and
does not create or replace a C84-A-E or C84-L1-L4 result. Every newly computed
order statistic, threshold distance and synthesis row is tagged
`POST_C84S_EXPLORATORY_DESCRIPTIVE` with `confirmatory_gate_changed=0`.

## Authoritative C84S Identities

```text
C84S final repository commit:
  2821c7099fc979672c4675e8c9ae54aa41ecd535

C84S V5 analysis-lock commit:
  2d03eb05e0cec352d08cdb6f48170be56876e77b

C84S V5 analysis-lock SHA-256:
  030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846

C84S authorization commit:
  47d405e96c1b0d3d2d35cd8bf5e14f95a3e933bb

C84S authorization-record SHA-256:
  3446e3562a8dd5db51c9f56a03765bf040f9678ee527ea13a4cf75e63dd575e1

C84S selection-freeze SHA-256:
  30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4

C84S scientific-result SHA-256:
  5590f85c3552ec0176a015e34296059a950dd2c5853a51aa140657cf53d79ee7

C84S result-manifest SHA-256:
  516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5

C84S result-identity SHA-256:
  9a2a1686c53409a2d5eb0d68c82f898406772c8e49e1f6e630e67c58f8ff9e44
```

## Frozen Scientific Population

```text
datasets:
  Lee2019_MI / Cho2017 / PhysionetMI

targets:
  22 / 20 / 76

source panels:
  A / B

training seeds:
  5 / 6

levels:
  0 / 1

target contexts:
  944

candidates/context:
  81

method-context rows:
  18,432
```

Dataset categories remain Lee C, Cho A and Physionet C. The common A and B
method intersections remain empty. Label B* remains absent for Lee, 8 for Cho,
and absent for Physionet. Cho's frozen level-specific B* values are FULL at
level 0 and 4 at level 1.

## C84A Read Boundary

C84A read only the C84S result JSON, result manifest, 18 frozen result tables,
selection/result lifecycle receipts and manifests, C84S committed reports, and
compact C80/C82/C83 evidence artifacts.

It did not read or invoke:

```text
EEG X
construction/evaluation label roots
target-logit or source-audit arrays
model checkpoints
Stage B or Stage C
selector formulas
Q0 chain builders
inference/max-T engines
training or forward
GPU
same-label oracle
```

No new p-value, simultaneous bound, selected candidate or scientific gate was
computed. Frozen p-values and decisions were copied by source key. A
level-specific Q2 simultaneous upper bound was not frozen, so C84A marks it
`NOT_FROZEN_AT_LEVEL_SCOPE` rather than reconstructing it.

## COTT Average-Tail Audit

The full-panel COTT mean Q1 effects remain positive in all three cohorts and
Q2 remains PASS in all three. Q1 remains FAIL in all three because the
registered worst-target floor fails.

```text
Lee:
  mean     0.148038
  median   0.161908
  q10      0.011322
  worst   -0.107873
  adverse targets 2
  floor breaches 1: target 8

Cho:
  mean     0.181005
  median   0.208972
  q10      0.014133
  worst   -0.305171
  adverse targets 2
  floor breaches 1: target 3

Physionet:
  mean     0.096640
  median   0.111489
  q10     -0.107382
  worst   -0.310447
  adverse targets 19
  floor breaches 9: 17 / 38 / 41 / 57 / 58 / 68 / 74 / 86 / 102
```

Lee target 8 is only 0.007873 below the -0.10 floor and is the only omission
that changes a frozen category, C to A. Cho target 3 is a much deeper breach.
Physionet's tail is distributed across multiple targets. This supports the
bounded description “positive average with non-robust target tail”; it does not
convert COTT into a Q1 success or a universal failure.

## MaNo Decision/Measurement Separation

Cho U11/MaNo remains the only dataset-level A method:

```text
mean regret:
  0.340079

Q1 mean:
  0.197703

Q1/Q2:
  PASS / PASS

worst target:
  -0.016765

mean Spearman:
  0.000960

top1 / top5 / top10:
  0.0500 / 0.15625 / 0.24375
```

All Cho MaNo selections are in the ERM regime and exactly match B1 selected
utility/regret in all 160 contexts. This is a descriptive concentration fact,
not a causal mechanism. The allowed compact tables do not include the full
81-candidate utility geometry, so near-optimal action density is not identified.

## Label-Frontier Decomposition

Lee B=8 and FULL have positive means and frozen max-T p-values below 0.05 but
fail the worst-target floor. Hence lower mean Q0 FULL regret does not satisfy
the compound actionability gate. Physionet B=8 and FULL fail max-T,
favorable-target and worst-target components. Cho B=8 and FULL pass all direct
components and larger-budget closure, yielding B*=8.

The level interaction is active: Lee COTT passes Q1/Q2 at level 0 but fails Q1
at level 1, while Cho's label frontier is FULL at level 0 and 4 at level 1.
These are observed effect-modifier patterns. C84A does not identify how support
deletion changes candidate-field geometry.

## LOTO And Heterogeneity

```text
Lee:
  21 / 22 categories preserved

Cho:
  20 / 20

Physionet:
  76 / 76
```

Target composition is mostly stable and is not the sole explanation of C84-D.
The A/C/C dataset mismatch, empty common method intersections and Lee level
heterogeneity already suffice.

The C82-to-C84 transport matrix retains nine fixed method identities across
BNCI2014_001 seeds 3/4 and the three C84 cohorts. It performs no pooling and no
new cross-study p-value.

## Decision-Theory Bridge

C84A distinguishes unrestricted optimal risk, registered-policy risk and the
policy approximation/optimization gap. C84S observes only fixed registered
policies under non-nested information classes. Therefore COTT or MaNo having
lower observed regret than Q0 in some rows does not establish that unlabeled
outputs Blackwell-dominate construction labels.

Q0 failure can reflect policy utilization, approximation and robust-tail
failure without implying that labels contain no value. No Blackwell, Le Cam,
minimax or CVaR theorem is proved.

## Theory Gaps And Future Review

Highest-priority gaps:

```text
formal information-experiment comparison
partial-identification/minimax target regret
average-risk versus worst-target/CVaR control
prospective active-versus-passive label policy
```

Additional candidate-geometry, heterogeneous-zoo and multi-paradigm work may
be reviewed, but each requires a new prospective protocol and untouched
population. C84A recommends review only. It does not create or authorize C85.

## Verification

Implementation/report commit:

```text
4076459996e2feecc9b7fa3aa6c036932f59f30e
```

Accepted Slurm regressions:

```text
focused job 898806:
  256 passed

C65 job 898807:
  867 passed, 1 skipped, 3 deselected

C23 job 898808:
  1,278 passed, 1 skipped, 3 deselected

full job 898809:
  2,202 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. The initial focused run under default
Python 3.9.13 is preserved as a rejected environment attempt; the exact locked
Python 3.13.7 rerun passed. `squeue` showed zero active C84/C85 jobs. `sacct`
was not used.

## Continuation Boundary

```text
C85:
  NOT AUTHORIZED

active acquisition:
  NOT AUTHORIZED

new datasets/model zoos:
  NOT AUTHORIZED

manuscript drafting/modification:
  NOT AUTHORIZED
```

A future step must begin with explicit PM protocol review and a new prospective
scope. It must not treat C84A's post-outcome summaries as independent evidence.
