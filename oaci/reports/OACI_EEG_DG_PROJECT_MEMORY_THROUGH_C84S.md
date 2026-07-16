# OACI / EEG-DG Project Memory Through C84S

## Current Scientific Gate

```text
C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

C84-L4
```

C84S V5 is the first complete scientific result from the harmonized
multi-dataset C84 field. Cho2017 is category A, Lee2019_MI and PhysionetMI are
category C, the cross-dataset A/B method intersections are empty, and level
heterogeneity is active. The passive common-grid label frontier exists at B=8
for Cho but is absent for Lee and Physionet, giving C84-L4.

## Authoritative Identities

```text
C84F complete-field manifest:
  cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8

C84SR3 repair protocol commit:
  91f984503fa84b53fae32948d0cf49e7ede12b8f

C84SR3 repair protocol SHA-256:
  5c783db9113697b2c710af4c1f1bafd66a3096be7a1b5cbac8aa03ca2a9c3080

lock-bound implementation commit:
  815d0ccd3f2ef245ea66c734165905d3a08ac105

C84S V5 lock commit:
  2d03eb05e0cec352d08cdb6f48170be56876e77b

C84S V5 lock SHA-256:
  030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846

authorization commit:
  47d405e96c1b0d3d2d35cd8bf5e14f95a3e933bb

authorization record SHA-256:
  3446e3562a8dd5db51c9f56a03765bf040f9678ee527ea13a4cf75e63dd575e1

authorization consumption SHA-256:
  ca362a16a49e349ea0945e64fb0636be3a59424cf8e146428e31d9e1c16b00de

selection-freeze SHA-256:
  30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4

scientific result SHA-256:
  5590f85c3552ec0176a015e34296059a950dd2c5853a51aa140657cf53d79ee7

result manifest SHA-256:
  516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5

result identity SHA-256:
  9a2a1686c53409a2d5eb0d68c82f898406772c8e49e1f6e630e67c58f8ff9e44
```

The complete report is `oaci/reports/C84S_OVERALL_REPORT.md` with a matching
machine-readable JSON report.

## Frozen Field And Scientific Population

```text
datasets:                  Lee2019_MI / Cho2017 / PhysionetMI
task:                      left_hand versus right_hand imagery
source panels:             A / B
training seeds:            5 / 6
levels:                    0 / 1
candidate units:           1,944
target subjects:           118
target trial rows:         9,621
target contexts:           944
candidates/context:        81
candidate-context slices:  76,464
```

Level 0 is the full fixed source panel. Level 1 is the fixed, target-independent
source-subject x `left_hand` support deletion. Target subject is the principal
within-dataset cluster. Panel, seed and level are repeated factors. Candidate
rows and Monte Carlo chains are not sample size.

## C84S V5 Lifecycle

Slurm job `898488` ran CPU-only on `cpu-high` with 48 CPUs and 128 GiB. It was
monitored with `squeue`; `sacct` was not used. Before authorization consumption,
the runtime byte-replayed all bound objects, 1,944 descriptors and 7,776
external artifacts totaling 48,072,941,176 bytes.

Stage A replayed the historical physical label views without loader calls or
label-row reload. Stage B used construction labels only and atomically froze
selection. Stage C received the evaluation seal only after that freeze. Exact
selection arithmetic:

```text
contexts:                       944
score rows:                 535,248
rank rows:                  535,248
fixed rows:                   4,720
Q0 shards:                      944
Q0 records:               8,750,000
Q0 sample digests:        1,093,750
Q0 chains/context-budget:      2,048
```

Exact result arithmetic:

```text
method-context rows:           18,432
target-level rows:              1,416
dataset Q1/Q2 rows:                18
level Q1/Q2 rows:                  36
LOTO rows:                        118
result tables:                     18
```

All table hashes and row counts match the atomic result manifest. Lee B32 is
input-unavailable and has no selection or result row; the primary grid remains
`[1,2,4,8,FULL]` for every dataset. Cho B32 remains secondary and operative.

## Dataset Results

### Lee2019_MI

Full-panel category C: no zero-label primary passes Q1. COTT/U13 has a positive
Q1 mean `0.148038`, max-T p `0.006760`, and 20/22 favorable targets, but its
worst target is `-0.107873`, below the registered `-0.10` floor. COTT passes
Q2. Level 0 COTT passes Q1+Q2 while level 1 fails Q1, creating active level
heterogeneity. LOTO category preservation is 21/22. No primary-grid B* exists.

### Cho2017

Category A: MaNo/U11 passes Q1 and Q2. Its Q1 mean is `0.197703`, max-T p is
`0.0000458`, 19/20 targets are favorable, and the worst target is `-0.016765`.
Its Q2 mean excess is `-0.167457` with simultaneous upper `-0.098524`. U11
remains the same supporting method in all 20 LOTO panels. The full-panel label
frontier is B*=8, while level-specific B* values are FULL and 4.

### PhysionetMI

Full-panel category C: no zero-label primary passes Q1. COTT has Q1 mean
`0.096640`, max-T p `0.028320`, and 57/76 favorable targets, but the worst
target is `-0.310447`. COTT passes Q2. LOTO category preservation is 76/76. No
primary-grid B* exists.

## Cross-Dataset And Objective Interpretation

```text
A_Lee:         none
A_Cho:         U11
A_Physionet:   none
A intersection: empty

B_Lee:         none
B_Cho:         U11
B_Physionet:   none
B intersection: empty
```

This A/C/C mismatch, empty common method set and Lee level disagreement force
C84-D. LOTO thresholds themselves pass. C84-L4 follows because Lee and
Physionet have no qualifying B*.

Ranking, regret and top-k remain distinct. COTT has positive mean Spearman in
all datasets but no dataset-level Q1 pass. Cho U11 passes Q1/Q2 with near-zero
mean Spearman. Physionet U11 has high top-k localization but no Q1 pass. No
measurement endpoint substitutes for decision performance.

## Protected State

```text
Stage-A label-loader calls:   0
target-label rows reloaded:   0
construction-label access:   1
evaluation-label access:     1, after selection freeze
training / forward / GPU:    0 / 0 / 0
same-label oracle:           0
C85 authorization:           false
```

One Stage-C SciPy warning records an undefined Spearman correlation for a
constant input. It produced no traceback and did not alter selection or the
scientific taxonomy.

## Historical Attempts

V3 job `897843` and V4 job `898192` remain immutable failed attempts. V3
stopped on a descriptor-compatibility gap. V4 stopped before one complete
selector context because Lee B32 was infeasible, with evaluation sealed; an
NFS cleanup error then masked the primary exception. Their authorizations and
partial roots were not reused. V5 reused only the immutable historical Stage-A
views through a no-loader replay.

## Claim And Authorization Boundary

C84S supports cohort-, target-, level- and objective-dependent selector
behavior on the harmonized binary MI task. It does not prove universal
zero-label impossibility, universal one-label sufficiency, universal EEG
external validity, deployability or causal mechanism.

C84S does not authorize C85, new methods, retuning, active acquisition, new
datasets, same-label oracle access or manuscript changes. The next permitted
activity requires separate PM direction and should begin with a complete
scientific/provenance audit and theory-gap analysis.
