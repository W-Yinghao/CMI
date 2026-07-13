# C80E Scientific Result Red Team

## Verdict

The frozen machine result at commit `a43aa27` passes the locked protocol and
maps to:

```text
C80-A_stable_low_regret_label_budget_frontier_across_training_seeds
```

This is the registered taxonomy label. The defensible claim is narrower: the
fixed Q0 construction-label policy attains a stable, materially positive
**source-relative regret-reduction frontier** at one label per class in the
existing seed-3/seed-4 fields. It is not evidence of low absolute regret,
reliable top-1 checkpoint recovery, few-label deployment, or external
generality.

The red team passed 32/32 registered integrity and claim-boundary checks with
zero blocking failures.

## Gate replay

```text
B*_seed3:                         1
B*_seed4:                         1
ordinal distance:                 0
cross-seed registered stability: PASS
seed3 B=1 regret reduction:       0.426093
seed4 B=1 regret reduction:       0.423742
seed4-minus-seed3 at B=1:        -0.002350
paired simultaneous 95% CI:      [-0.072244, 0.067543]
positive targets at every budget: 8 / 8 for both seeds
catastrophic targets:             0
B=1 maxT p, each seed:            0.042802
```

All five registry paths executed unconditionally. Target 4 is absent from all
primary rows, the same-label oracle remained closed, and all 22 frozen result
artifacts replay by hash.

## Mandatory caveats

### The gate is source-relative

At B=1, expected standardized regret remains `0.353383` for seed 3 and
`0.373705` for seed 4. Top-1 hit rates are only `0.037842` and `0.038391`.
The source baseline is unusually weak: mean standardized regret is `0.779476`
and `0.797447`, worse than the corresponding random-policy expectations
`0.481982` and `0.497584`. Therefore C80-A is driven by large improvement over
the registered source baseline, not by near-oracle checkpoint identification.

At B=1, top-5 is `0.159210`/`0.164825` and top-10 is
`0.280792`/`0.294128`. These exceed the 81-candidate random baselines but remain
far from deterministic recovery.

### The minimal budget is small-target sensitive

Every one of the 16 seed-by-left-out-target sensitivity analyses changes B*
from 1 to 2 or 4. This does not alter the locked eight-target primary taxonomy,
but it prevents a strong universal or population-level claim about one label
per class. The B=1 exact maxT p-value is also close to 0.05.

The cross-seed heterogeneity gate passes because the mean difference is within
the locked margin and the simultaneous interval contains zero. The interval is
not an equivalence interval contained within the margin, so the result must not
be described as statistical equivalence of the two seeds.

### Curves and moderation are heterogeneous

Four seed-3 targets and three seed-4 targets have a nonmonotone target-level
regret curve, generally because FULL is not always better than budget 32. The
registered closure rule, rather than smoothing or interpolation, remains
authoritative.

S3 does not identify stable moderation. Raw candidate count is constant at 81,
so its correlation is undefined. Effective multiplicity correlations change
from mostly negative on seed 3 to positive on seed 4; top-gap correlations
change from positive on seed 3 to negative on seed 4. S3 remains descriptive
and does not rescue H2/H2R or alter B*.

## Scope boundary

The result uses the same targets, raw trials, candidate fields, and fixed split
already observed in C79. It is retrospective existing-field design evidence.
It does not establish a universal minimum label budget, a new-subject or
target-population result, external validation, active acquisition value, a
representation mechanism, or checkpoint deployability.

Gate: `C80E_SCIENTIFIC_RESULT_RED_TEAM_PASSED_WITH_MANDATORY_CAVEATS`.
