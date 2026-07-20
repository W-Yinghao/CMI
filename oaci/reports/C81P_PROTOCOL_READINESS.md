# C81P - Frozen-Field Literature Baseline Comparison Readiness

## Final Gate

```text
C81_AAAI_BASELINE_COMPARISON_PROTOCOL_LOCKED_READY_FOR_PI_AUTHORIZATION
```

Primary taxonomy:

```text
C81P-A_frozen_field_baseline_comparison_protocol_locked_complete
```

C81P is complete protocol/readiness work. It is designed after C79/C80 and is
prospective only to the new baseline computations. It is an outcome-informed
comparison on existing fields, not independent confirmation, external
validation, or evidence from new subjects or datasets.

No real C81 baseline score, candidate ranking, evaluation-label value, or
same-label-oracle value was read or computed.

## Operative Objects

```text
protocol commit:          16a0d2eba4715a1cec78da6a79a182fd416a6629
protocol SHA-256:         cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee
method registry SHA-256:  ef48ecf7fcc55188b78b0878d86f07f6239fe4f6c88bbc854829b3a1c7a1a120
implementation commit:    d17ffa62a63b929d36d03f74e4ce79794cd9601b
analysis lock commit:     541651c2ee3343c12d374a7322c91181a860a2c9
analysis lock SHA-256:    b383707f58063c10f719194a995ab34094f6dcefe08c1e71837644db83dc94f1
pre-execution red team:   43 / 43 PASS
C81E authorization:      absent
```

The lock binds the two implementation blobs, method registry, 11 frozen
seed-3/seed-4 field and view objects, score directions, priors, ties,
representatives, inference, taxonomy, two-stage result schema, and fail-closed
authorization boundary.

## C80 Input Replay

The accepted C80 field and decision frontier were replayed without recomputing
a baseline:

```text
C80 result artifact hashes:       22 / 22 PASS
field/view manifest objects:      11 / 11 PASS
frontier rows:                    14 / 14 PASS
top-k and coverage rows:          14 / 14 PASS
target-level rows:               224 / 224 PASS
leave-one-target rows:            16 / 16 PASS
target 4 primary rows:             0
same-label-oracle access events:   0
```

The operative C80 analysis lock is `0797599`, with SHA-256
`2149895865bd44b4ab8358c76848bb6774abb59d4a203b261864be0ec599ff62`.
The earlier complete lock `f19acd8` remains historical and superseded after a
pre-evaluation descriptor ABI failure. The additive repair changed no
scientific object and read zero evaluation labels.

The fixed labeled comparators preserve the accepted C80 context: B*=1 for both
seeds in the full panel, substantial absolute regret, approximately 3.8% top-1
localization, and 16/16 leave-one-target analyses moving B* to 2 or 4.

## Frozen Comparison

```text
training seeds:       [3, 4]
primary targets:      [1, 2, 3, 5, 6, 7, 8, 9]
levels:               [0, 1]
candidates/context:   81 = 1 ERM + 40 OACI + 40 SRC
primary contexts:     32
primary candidates:   2,592
principal cluster:    target
seed role:            paired training factor
```

Every method selects within the same 81-candidate context. Target 4 is
engineering-only. Trial rows, checkpoints, candidate pairs, and Monte Carlo
draws are not scientific replicates.

The registry contains 34 methods: 28 feasible controls, comparators, selectors,
or diagnostics; five methods excluded because required frozen inputs are
absent; and one oracle-best denominator that is not a selector. The real
adapter exposes 19 registered score paths plus fixed C80 Q0 comparators.

Primary representatives were frozen before outcomes:

```text
R0  uniform random
R1  ERM anchor deployment default
R2  source-validation balanced accuracy
R3  ATC
R4  normalized nuclear norm
R5  MaNo
R6  COTT
R7  SND
R8  ALine
R9  unavailable: no faithful training-free importance-weighting member
R10 frozen Q0 B=1
R11 frozen Q0 FULL
```

True Source-LODO is unavailable because the frozen artifacts do not contain the
required source-held-out retraining folds, so the predeclared fallback R2 is
used. IWCV, DEV, and IW-GAE remain excluded because their required density
ratio, domain-discriminator, or fitted group-weight inputs are absent and new
training/fitting is outside C81. The historical F2 aggregate is not promoted to
a selector because no execution-bound per-candidate descriptor exists.

## Information and View Contract

The comparison spans `I0`, `IS`, `IU`, `ISU`, and `ILc`; `IOr` is a
descriptive denominator only. IS methods receive source labels and source
outputs. IU methods receive target-unlabeled outputs or frozen features. ISU
methods may calibrate on source labels and consume target-unlabeled objects.
The Q0 comparators reuse frozen C80 construction-label outputs.

Selection artifacts are frozen and content-addressed before the evaluation
stage can open evaluation labels. The same-label oracle is unreachable. Trial
ID and row order are keys for joining, splitting, and dependence only, never
predictive features.

## Inference and Taxonomy

Q1 tests material zero-label regret improvement over the locked strict-source
representative. Q2 tests zero-label noninferiority to Q0 B=1 using the locked
0.05 margin. Both use shared-target exact sign-flip max-T inference over 256
sign patterns, at least 6/8 favorable targets, and registered catastrophic
target rules.

LOTO stability requires matching full-panel seed categories and at least 12/16
seed-by-left-out-target panels retaining the full-panel category. The exhaustive
precedence is blocker C81-E, heterogeneity C81-D, then stable C81-A, C81-B, or
C81-C. The narrative cannot select a better-performing secondary family member
after results are known.

## Synthetic Calibration

Synthetic/schema-only calibration used no real-field score:

```text
registered scenarios:             13 / 13 PASS
family-wise calibration:           2 / 2 PASS
pair/trial dependence checks:      2 / 2 PASS
noninferiority calibration:        3 / 3 PASS
exact max-T observed FWER:          0.050781
unadjusted-test negative control:   0.437500
```

The suite exercises local confidence nontransport, ALine success and pair
dependence, SND association without actionability, ATC/DoC shift behavior,
COTT prior mismatch, one-label low regret with poor top-1, target
heterogeneity, and pseudoreplication traps.

## Regression

Accepted Slurm CPU regressions ran on clean commit
`e347a06edf9fdcabd999cb848301d7d6b025c36c`:

```text
focused C81P (894763):  43 passed
C65-C81P (894764):     412 passed, 1 conditional skip, 3 deselected
C23-C81P (894765):     823 passed, 1 conditional skip, 3 deselected
full OACI (894766):  1,747 passed, 1 conditional skip, 3 deselected
accepted stderr bytes: 0 for all four jobs
```

The conditional skip is the finalized C78F guard. The three deselections are
historical C79P preauthorization-state tests and conceal no C81 path. All
superseded, cancelled, malformed-command, wrong-SHA, and dirty-worktree attempts
remain in `c81p_tables/regression_attempt_ledger.csv`.

## Authorization Boundary

C81P does not authorize C81E. Under the policy committed at `3d9dd76`, a direct
PI statement naming C81E is sufficient; no token or repeated hash recital is
required. The server record must bind that statement to the unique current C81
protocol and execution lock above.

Future C81E remains read-only existing-field analysis. It does not authorize
training, forward/re-inference, GPU, target-4 primary use, same-label oracle,
BNCI2014_004, seed 5, active acquisition, new features/kernels/models, or
manuscript experiments.
