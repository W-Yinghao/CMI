# C82 Post-C81 Frozen-Selection Baseline Recovery

## Final Gate

```text
C82-D_zero_label_comparison_training_seed_method_identity_or_target_heterogeneous
```

C82 is a new post-C81-outcome-access result identity. It does not repair or
replace the historical C81 gate:

```text
C81-E_protocol_input_implementation_or_provenance_blocker
```

The latest pre-C82 scientific result, C80E, also remains unchanged.

## Scientific Status

C82 reused the exact C81 selection frozen before C81 evaluation access and
scored it under a new, explicitly post-outcome-access protocol. The analysis is
prospective only to the new C82 computation and result freeze. It uses the same
seed-3/seed-4 candidate field, targets, and raw trials; it is not independent
confirmation, new-subject evidence, external validation, or target-population
replication.

The registered question was whether one fixed zero-target-label selector could:

1. materially improve standardized regret over the fixed strict-source primary
   comparator in both training seeds;
2. be noninferior to the frozen Q0 one-construction-label-per-class comparator;
3. preserve the same method-level conclusion across seeds and at least 12 of 16
   leave-one-target panels.

No registered method satisfied that joint requirement.

## Execution And Identity

```text
protocol commit:               8b0df50b3707dbb3af4a459b6dc6de36c97d562f
protocol SHA-256:              9f58c7a8e6b495a6d8f510c0d72d24ede4485908ef94bc078abe8f124b03a8f3
analysis lock commit:          6c6739c61d362bc33df6d8b016e4cda724772a62
analysis-lock SHA-256:         d5de6d6ff242b9f3d7f9c318cbdd6e1e16c509060bc14cca59292b738a75f5ce
authorization commit:          5644157
scientific job:                895214
job state / exit:              COMPLETED / 0:0
runtime / allocation:          4 s / 48 CPU / 96 GiB / GPU 0
result-freeze commit:          ce0564d
scientific red-team commit:    61b3fe2
```

The first scheduler submission, job `895213`, was rejected in zero seconds by a
mistyped shell-level HEAD guard before the Python adapter started. It opened no
payload or view and did not consume authorization. The attempt and additive
wrapper-only correction are retained. Job `895214` was the only scientific
execution.

The successful job consumed authorization before opening the frozen selection
or evaluation descriptors, replayed the exact selection, opened 16 registered
target-evaluation views containing 4,746 rows, and atomically published the
complete result.

```text
selection manifest self SHA-256:
  4677ed3aba7758ea0008c2093b44d6fb81d425930727e5941950179737ebd519

selection payload SHA-256:
  1ed893acd9190914eb4cb122f3ef26bc1e2355c4103894b816894bd264669257

result SHA-256:
  d8060e6636adf7fcca7a0ace0e47bb7043676b7681569e09fb8705dcb8d5a8b7

artifact-manifest SHA-256:
  910e2ff1d8445dae262be82d417140cd44fc48be1306f2bbe5a439ec3549f0a2

method-context rows: 672 / 672
registered tables:    23 / 23
artifact hashes:      23 / 23
artifact row counts:  23 / 23
```

## Registered Decision

| Object | Seed 3 | Seed 4 | Cross-seed result |
|---|---|---|---|
| Q1: material improvement over strict source | COTT (`U13`) only | none | no common method |
| Q2: noninferior to Q0 B=1 | none | none | no common method |
| Seed category | B | C | heterogeneous |
| A-method intersection | empty | empty | empty |
| B-method intersection | `U13` vs none | none | empty |
| LOTO panels preserving required category and method | 0/8 | 7/8 | 7/16, below 12/16 |

The exact taxonomy precedence therefore selects C82-D before C82-A/B/C.

## Q1: Zero Label Versus Strict Source

The primary strict-source comparator was source-validation balanced accuracy
(`S1`). Its mean standardized regret was high on both fields:

```text
S1 regret:
  seed 3: 0.779476
  seed 4: 0.804823
```

COTT (`U13`) was the only zero-label primary representative to pass Q1, and it
did so only on seed 3:

| Quantity | Seed 3 | Seed 4 |
|---|---:|---:|
| COTT standardized regret | 0.338641 | 0.465335 |
| Improvement over S1 | 0.440835 | 0.339488 |
| Simultaneous lower bound | 0.283446 | 0.182099 |
| max-T p | 0.015564 | 0.101167 |
| Favorable targets | 8/8 | 7/8 |
| Worst target effect | 0.166885 | -0.076885 |
| Q1 decision | PASS | FAIL |

The positive mean direction is concordant, but the registered inferential and
target-consistency gate is not training-seed robust. ATC, NuclearNorm, MaNo,
SND, and ALine did not pass Q1 on either seed.

## Q2: Zero Label Versus One Construction Label

The frozen Q0 B=1 comparator retained its C80 values:

```text
Q0 B=1 standardized regret:
  seed 3: 0.353383
  seed 4: 0.373705
```

No zero-label primary representative passed the locked noninferiority rule in
either seed. COTT was numerically closest:

| Quantity | Seed 3 | Seed 4 |
|---|---:|---:|
| COTT minus Q0 B=1 mean regret | -0.014743 | 0.091630 |
| Simultaneous upper bound | 0.144528 | 0.250901 |
| max-T p | 0.517510 | 1.000000 |
| Favorable targets | 6/8 | 4/8 |
| Q2 catastrophic excess targets | 2/8 | 3/8 |
| Q2 decision | FAIL | FAIL |

The seed-3 mean alone would make COTT appear slightly better than Q0 B=1, but
the registered simultaneous target-cluster result does not support
noninferiority. This distinction is central: a favorable pooled mean is not a
stable target-level decision guarantee.

## Q3: Objective Dependence

Decision conclusions differed across regret and localization endpoints.

| Method and seed | Top-1 | Top-5 | Top-10 | Regret Q1 |
|---|---:|---:|---:|---|
| COTT seed 3 | 0.1250 | 0.3125 | 0.3750 | PASS |
| COTT seed 4 | 0.0000 | 0.2500 | 0.3125 | FAIL |
| S1 seed 3 | 0.0000 | 0.0000 | 0.0000 | comparator |
| S1 seed 4 | 0.0000 | 0.0000 | 0.0000 | comparator |
| Q0 B=1 seed 3 | 0.0378 | 0.1592 | 0.2808 | C80 comparator |
| Q0 B=1 seed 4 | 0.0384 | 0.1648 | 0.2941 | C80 comparator |

COTT's seed-3 exact-best localization was better descriptively than Q0 B=1,
but it collapsed to zero top-1 on seed 4. Top-k behavior cannot override the
registered regret, max-T, noninferiority, or stability decisions.

## Q4: Training Seed And Target Composition

The full panel classified seed 3 as B because COTT passed Q1 but no method
passed Q2. Seed 4 classified as C because no method passed Q1.

Every seed-3 leave-one-target panel retained a qualitative B category only via
a different supporting method; none preserved the full-panel COTT identity.
Seven of eight seed-4 panels preserved category C, while leaving out target 2
changed the category to B. Therefore only 7 of 16 method-aware panels passed,
well below the registered 12/16 threshold.

This is direct evidence that the comparative conclusion is sensitive to both
training seed and target composition under the frozen field. Seed is a paired
training factor and targets are the scientific clusters; the 672 rows are not
672 independent observations.

## Q5: Information-Class Summary

The registered descriptive best method within each information class was:

| Information class | Seed 3 best (regret) | Seed 4 best (regret) |
|---|---|---|
| I0 fixed no-information controls | midpoint OACI `B4O` (0.366854) | final OACI `B2` (0.350480) |
| IS source-only methods | source NLL `S2` (0.745842) | source NLL `S2` (0.751183) |
| IU/ISU zero-label methods | COTT `U13` (0.338641) | COTT `U13` (0.465335) |
| ILc construction-label curve | Q0 FULL (0.082794) | Q0 FULL (0.110667) |

The ordering is not monotone across I0, IS, and IU/ISU, and the best fixed
no-information default outperformed COTT on seed 4. Independent construction
labels, especially at larger budgets, produced much lower regret on this field.
This is a field- and policy-specific comparison, not an information-theoretic
ordering or impossibility theorem.

## Measurement Versus Decision

COTT had the strongest positive registered zero-label ranking relationship:

```text
mean within-context Spearman:
  seed 3: 0.276605
  seed 4: 0.184232

mean pairwise ordering accuracy:
  seed 3: 0.600160
  seed 4: 0.568720
```

Those measurement effects coexist with Q1 success only on seed 3 and Q2 failure
on both seeds. They do not substitute for decision utility.

The U16 Accuracy-on-the-Line diagnostic was uniformly poor and remained
secondary:

```text
incremental R2 mean / range:
  seed 3: -0.667244 / [-1.311624, -0.212906]
  seed 4: -0.769685 / [-1.461965, -0.311497]
```

LORO was prospectively removed from operative C82 inference because C81 did not
define a coherent mixed-81-candidate estimand. C82 makes no cross-regime
selector-transport claim.

## Regime Composition

COTT selected:

```text
seed 3: 62.50% SRC, 37.50% OACI
seed 4: 68.75% SRC, 25.00% OACI, 6.25% ERM
```

This is selection composition only. It is not an SRC rescue, generator-quality
claim, or evidence that a regime itself is deployable.

## Accepted Conclusions

The registered C82 audit supports the following narrow conclusions:

1. COTT materially improved over the strict-source primary comparator on seed
   3, but the exact Q1 gate did not pass on seed 4.
2. No registered zero-label primary representative was noninferior to frozen Q0
   B=1 under the simultaneous target-cluster rule on either seed.
3. No one fixed zero-label method supported a common A or B category across
   both seeds.
4. The comparative category was highly leave-target sensitive, preserving the
   required method-aware result in only 7/16 panels.
5. Measurement association, regret improvement, noninferiority, top-k
   localization, and target robustness remain distinct evidence levels.

## Claims Not Supported

C82 does not establish:

```text
universal zero-label selector impossibility
universal one-label sufficiency
external or target-population validity
new-subject or new-dataset replication
causal representation mechanism
cross-regime selector transport
SRC or OACI rescue
deployability
active-acquisition optimality
```

Target 4 contributed zero primary rows. The same-label oracle remained closed.
Selection was not recomputed. Construction-label content was not reopened.
Training, forward, re-inference, and GPU work were zero.

## Control State

C82 ends for PM review at C82-D. It does not authorize C83, another seed,
BNCI2014_004, target 4, same-label oracle analysis, active acquisition, new
methods, new training, or manuscript experiments.
