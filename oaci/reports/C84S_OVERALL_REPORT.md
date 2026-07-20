# C84S Multi-Dataset Scientific Execution Overall Report

## 1. Final Decision

C84S V5 completed the registered three-dataset scientific analysis and
atomically froze one primary gate and one label-frontier tag:

```text
primary gate:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

label-frontier tag:
  C84-L4
```

The result is heterogeneous, not a universal success or failure of zero-label
selection. Cho2017 is category A, whereas Lee2019_MI and PhysionetMI are
category C under the complete registered full-panel gates. The same-method
cross-dataset A and B intersections are empty. A level-specific Lee result is
positive, so averaging levels would hide registered heterogeneity.

## 2. Scientific Scope And Claim Boundary

C84 evaluates a harmonized binary left-hand versus right-hand motor-imagery
task in three external cohorts:

```text
datasets:             Lee2019_MI / Cho2017 / PhysionetMI
target subjects:      22 / 20 / 76 = 118
source panels:        A / B
training seeds:       5 / 6
levels:               0 / 1
target contexts:      944
candidates/context:   81
candidate units:      1,944
```

Level 0 uses the complete fixed source panel. Level 1 uses the prospectively
registered fixed source-subject x `left_hand` support deletion. Each target is
the principal within-dataset scientific cluster; panel, seed and level are
repeated factors. Candidate rows and 2,048 Q0 chains are not independent
scientific samples. No pooled three-dataset p-value is used.

C84 is external-cohort evidence for this harmonized binary task. It is not an
exact numerical replication of the four-class BNCI2014_001 field, does not
establish universal EEG validity, and does not authorize a claim of universal
zero-label impossibility or universal one-label sufficiency.

## 3. Protocol, Lock, Authorization, And Result Identity

```text
C84SR3 repair protocol commit:
  91f984503fa84b53fae32948d0cf49e7ede12b8f

C84SR3 repair protocol SHA-256:
  5c783db9113697b2c710af4c1f1bafd66a3096be7a1b5cbac8aa03ca2a9c3080

lock-bound implementation commit:
  815d0ccd3f2ef245ea66c734165905d3a08ac105

C84S V5 analysis-lock commit:
  2d03eb05e0cec352d08cdb6f48170be56876e77b

C84S V5 analysis-lock SHA-256:
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

result artifact-manifest SHA-256:
  516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5

result identity SHA-256:
  9a2a1686c53409a2d5eb0d68c82f898406772c8e49e1f6e630e67c58f8ff9e44
```

The complete frozen C84F field manifest remains
`cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8`.
No candidate, model state, trial identity, method, threshold, budget or
inference rule changed during C84S.

## 4. Execution Lifecycle

Slurm job `898488` ran on `cpu-high` with 48 CPUs, 128 GiB, zero GPUs and a
48-hour limit. It was monitored with `squeue`; `sacct` was not used. The
post-consumption lifecycle took 2,975.920 seconds:

| Stage | Role | Seconds | Exit |
|---|---|---:|---:|
| A | Immutable historical label-view replay | 0.194 | 0 |
| B | Construction-only selection and freeze | 2,297.316 | 0 |
| C | Held evaluation and registered inference | 678.291 | 0 |

Before authorization consumption, the runtime replayed the V5 lock, all bound
repository objects, the full-scale synthetic calibration, 1,944 field
descriptors and 7,776 external selection artifacts totaling 48,072,941,176
bytes. The output root did not exist before replay.

Stage A replayed the immutable V3 label-view handoffs with zero loader calls
and zero target-label row reloads. Stage B received only the construction
handoff. The evaluation descriptor remained sealed until the complete Stage-B
manifest was published. Stage C then received the immutable selection freeze
and the evaluation seal, with no callable able to mutate selection.

Slurm stderr, Stage-A stderr and Stage-B stderr are empty. Stage C emitted one
SciPy `ConstantInputWarning` for an undefined Spearman correlation on a
constant score vector. The registered applicability/null handling retained an
undefined measurement value; the warning did not alter selection, regret,
Q1/Q2 or taxonomy. No traceback or runtime failure marker occurred.

## 5. Exact Selection Freeze

Stage B atomically froze:

```text
contexts:                         944 / 944
candidate-score rows:        535,248 / 535,248
candidate-rank rows:         535,248 / 535,248
fixed-default rows:            4,720 / 4,720
Q0 context shards:               944 / 944
Q0 finite/FULL records:    8,750,000 / 8,750,000
Q0 sample-digest rows:     1,093,750 / 1,093,750
Stage-B Q0 regime rows:       15,648 / 15,648
Stage-B Q0 coverage rows:      5,216 / 5,216
```

The primary Q0 grid `[1,2,4,8,FULL]` is unchanged for every dataset. Lee uses
secondary B16 only; Cho uses secondary B16/B32; Physionet has no secondary
budget. Lee B32 remains
`INPUT_UNAVAILABLE_NO_SELECTION_OR_RESULT_ROW` because every Lee construction
cell has 25 labels per class. It has no Q0 record and no method-context row.
No replacement sampling, FULL substitution or target-specific budget repair
was used.

## 6. Exact Held-Evaluation Freeze

Stage C atomically validated and published 18 registered tables:

```text
method-context rows:             18,432 / 18,432
target-level method rows:         1,416 / 1,416
dataset Q1/Q2 rows:                  18 / 18
level-specific Q1/Q2 rows:           36 / 36
panel/seed rows:                      18 / 18
LOTO rows:                           118 / 118
primary label-frontier rows:          15 / 15
Q0 regime rows:                   12,816 / 12,816
Q0 Monte Carlo rows:              4,272 / 4,272
```

Every file row count and SHA-256 matches
`C84S_RESULT_ARTIFACT_MANIFEST.json`. All methods shown in the 18,432-row
table have coverage 1.0. Q0 chains are integrated within context and never
enter target-cluster sample size.

## 7. Dataset-Level Q1 And Q2 Results

The six zero-label primaries are U5 NuclearNorm, U7 ATC, U11 MaNo, U13 COTT,
U14 SND and U15 Agreement-on-the-Line. S1 is strict-source validation. Q1 is
material improvement over S1; Q2 is noninferiority to Q0 B=1.

| Dataset | Category | Q1 methods | Q1+Q2 methods | Registered interpretation |
|---|---|---|---|---|
| Lee2019_MI | C | none | none | no full-panel Q1 pass; hidden level-specific pass exists |
| Cho2017 | A | U11 | U11 | MaNo passes Q1 and Q2 under every registered component |
| PhysionetMI | C | none | none | no full-panel Q1 pass; COTT passes Q2 only |

The most informative frozen Q1/Q2 rows are:

| Dataset / method | Q1 mean | max-T p | Favorable | Worst target | Q1 | Q2 excess | Q2 upper | Q2 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Lee / U13 COTT | 0.148038 | 0.006760 | 20/22 | -0.107873 | fail | -0.155298 | -0.085920 | pass |
| Cho / U11 MaNo | 0.197703 | 0.000046 | 19/20 | -0.016765 | pass | -0.167457 | -0.098524 | pass |
| Cho / U13 COTT | 0.181005 | 0.002869 | 18/20 | -0.305171 | fail | -0.150759 | -0.081826 | pass |
| Physionet / U13 COTT | 0.096640 | 0.028320 | 57/76 | -0.310447 | fail | -0.065032 | -0.030233 | pass |

Lee COTT misses Q1 only because its worst target is below the registered
`-0.10` floor. Physionet COTT also has a positive mean, a family-wise p-value
below 0.05 and exactly 75% favorable targets, but its worst target is far below
the floor. These are target-robustness failures, not negative average effects.

COTT passes Q2 in all three datasets, but Q2 alone cannot create an A method:
Q1 and Q2 must both pass in the same dataset and the same method must recur
across datasets. Only Cho U11 satisfies both gates. Therefore:

```text
A_Lee intersection member:         none
A_Cho member:                      U11
A_Physionet member:                none
A cross-dataset intersection:      empty

B_Lee member:                      none
B_Cho member:                      U11
B_Physionet member:                none
B cross-dataset intersection:      empty
```

## 8. Regret, Top-k, And Measurement Separation

Selected mean regret illustrates the direction without replacing the gates:

| Dataset | S1 | U11 | U13 | Q0 B=1 | Q0 FULL |
|---|---:|---:|---:|---:|---:|
| Lee | 0.425163 | 0.300787 | 0.277125 | 0.432423 | 0.286498 |
| Cho | 0.537782 | 0.340079 | 0.356778 | 0.507536 | 0.209206 |
| Physionet | 0.446184 | 0.381123 | 0.349544 | 0.414576 | 0.376618 |

Lower aggregate regret is insufficient when a registered target, panel/seed,
level or max-T component fails. Top-k also gives a distinct view. For example,
Physionet U11 has top-1/top-5/top-10 rates of
`0.2286/0.4836/0.5707` but does not pass Q1. Cho U11 passes Q1 with much lower
`0.0500/0.1563/0.2438` localization rates.

Rank association likewise does not substitute for regret. COTT mean Spearman
is `0.2835`, `0.1798` and `0.1441` in Lee, Cho and Physionet, respectively,
yet COTT has no dataset-level Q1 pass. Conversely, Cho U11 passes Q1/Q2 with
mean Spearman approximately `0.0010`. Performance-estimation MAE remains null
for methods where it is semantically inapplicable; no zero was manufactured.

## 9. Level, Panel/Seed, And Target-Composition Stability

The full result has `LEVEL_HETEROGENEITY = true`. In Lee:

```text
U13 level 0: Q1 pass, Q2 pass
U13 level 1: Q1 fail, Q2 pass
U11 level 0: Q1 fail, Q2 pass
U11 level 1: Q1 fail, Q2 fail
```

Thus Lee contains a hidden level-0 A result that cannot be promoted to a
full-panel Lee A result. This alone activates the registered heterogeneity
precedence even before considering the A/C/C dataset-category mismatch.

Panel/seed directional checks pass for the principal positive rows where
reported: Cho U11 is Q1-positive and Q2-within-margin in all four panel x seed
cells; COTT is directionally Q1-positive and Q2-within-margin in all four cells
for Lee, Cho and Physionet. The full composite gates still retain their target
robustness and level requirements.

LOTO method/category preservation is:

```text
Lee:        21 / 22, threshold 17
Cho:        20 / 20, threshold 15
Physionet:  76 / 76, threshold 57
```

Lee changes from C to A only when target 8 is omitted. Cho retains U11 as its
same supporting method in every LOTO panel. These LOTO thresholds pass; they
do not erase cross-dataset or level heterogeneity.

## 10. Passive Label-Budget Frontier

The primary common budget result is:

| Dataset | B* | Level-0 B* | Level-1 B* | LOTO preservation |
|---|---|---|---|---:|
| Lee | none | none | none | 21/22 |
| Cho | 8 | FULL | 4 | 19/20 |
| Physionet | none | none | none | 76/76 |

Cho B=8 and FULL pass the registered larger-budget closure. Cho's level-specific
frontiers differ, so label-budget level heterogeneity is disclosed. Lee and
Physionet have no qualifying primary-grid B*. Because at least one dataset has
no B*, the exact registered cross-dataset tag is `C84-L4`; the existence of a
Cho frontier cannot be generalized to the other cohorts.

## 11. Historical Attempts And Repair Provenance

Historical V3 job `897843` and V4 job `898192` remain immutable failed
attempts. V3 failed on a field-descriptor compatibility gap before selector
scoring. V4 consumed its authorization and entered construction-only Stage B,
then stopped before one complete selector context because Lee B32 was
physically infeasible; evaluation remained sealed. Its NFS cleanup error
masked the primary exception. C84SR3 repaired budget availability and atomic
failure semantics prospectively, created V5, and required the fresh
authorization used by job `898488`.

No V3/V4 partial selection artifact, consumed authorization or failed root was
reused as a V5 selection result. Only the immutable historical Stage-A label
views were replayed without a loader call.

## 12. Protected Counters

```text
Stage-A label-loader calls:       0
target-label rows reloaded:       0
construction-label access:       1
evaluation-label access:         1, after selection freeze
selector contexts frozen:        944
scientific rows frozen:           18,432
training:                         0
forward:                          0
GPU:                              0
same-label oracle:                0
C85 authorization:               false
```

Construction labels were used only for Q0 selection. Evaluation labels became
available only after the complete selection freeze. B5 remained an evaluation
ceiling and never entered Stage B.

## 13. Scientific Interpretation

The registered information hierarchy does not recur uniformly across the
three cohorts. A fixed zero-label method, MaNo/U11, matches the B=1 frontier in
Cho under the full Q1/Q2 gate. Lee and Physionet have positive aggregate COTT
effects and COTT Q2 passes, but target-level robustness prevents Q1. Lee also
changes qualitatively across source-support levels. Passive label-budget
actionability appears at B=8 only in Cho and is absent on the primary grid in
Lee and Physionet.

The supported conclusion is therefore cohort-, target-composition-, level- and
decision-objective heterogeneity. The result does not show that MaNo or COTT is
generally effective, that zero-label selection is generally impossible, that
one label is generally sufficient, or that a fixed passive label budget is
deployable across EEG datasets.

## 14. Verification And Reporting

The initial scientific report was committed at
`c2f92f65c310aa895f6a2bcb060df07a6fbf475b`. Scientific red-team passed
64/64 checks and final-report red-team passed 72/72 checks.

Post-execution CPU regressions used the locked
`c84c-eeg2025-v3-exact` environment:

| Suite | Slurm job | Result | Pytest time | stderr |
|---|---:|---|---:|---:|
| focused C84 | 898593 | 367 passed | 75.32 s | 0 bytes |
| C65 cumulative | 898594 | 853 passed, 1 skipped, 3 deselected | 108.64 s | 0 bytes |
| C23 cumulative | 898619 | 1,264 passed, 1 skipped, 3 deselected | 165.47 s | 0 bytes |
| full OACI | 898621 | 2,188 passed, 1 skipped, 3 deselected | 488.13 s | 0 bytes |

The skip is the finalized C78F field test. The three deselections are the
established C79 tests whose premise is that a later authorization record is
absent. Two initial C23/full submission pairs (`898595/898596` and
`898615/898616`) were rejected by the regression script before pytest because
the report worktree was transiently uncommitted or not yet visible as clean on
the compute node. They are preserved as failed preflight attempts. Read-only
job `898617` then confirmed exact `HEAD == origin/oaci` and a clean compute-node
worktree before the accepted retries.

Tracked Git hygiene passed: no tracked file exceeds 50 MiB, and no raw EEG,
checkpoint, weight, optimizer or NumPy cache artifact entered Git. The final
C84S external root has no residual staging directory and no traceback.

## 15. Authorization Boundary

C84S completion does not authorize C85, new selectors, retuning, additional
datasets, active acquisition, oracle access or manuscript changes. The next
valid action is a complete scientific/provenance audit and theory-gap review
under separate PM direction.
