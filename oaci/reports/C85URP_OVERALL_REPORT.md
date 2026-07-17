# C85URP Overall Report

## Disposition

```text
C85U_HELD_EVALUATION_CANDIDATE_UTILITY_RECONSTRUCTION_LOCKED_READY_FOR_PI_AUTHORIZATION
```

C85URP is complete. It produced the prospective protocol, isolated U1/U2
implementation, metadata-only frozen-input registry, shadow validation, and
one C85U execution lock. It did not execute real reconstruction.

## Locked Identities

```text
protocol commit:
  ebe158c9e929f67423a9ebdc3cea7c6ea5c16c9a

protocol SHA-256:
  c9ed7081cf8cb1a6c8a05181d1660da2015b4e1716a05c8916f7fe5b09efc160

implementation commit:
  df100e2e77c5749030e2931bb7752973258823bd

execution-lock commit:
  3b4fa48ee2d4f75ff8ba2191dc7d8593237dc82f

execution-lock SHA-256:
  923c6bee2171f0bedcc3f883058759d368bdb49eb272cbbfa80974e98b632fe1

runtime-bound repository objects:
  45
```

## Frozen Scope

```text
datasets:                     3
field descriptors:        1,944
target artifacts:         1,944
target-artifact bytes:     48,018,748,054
target contexts:             944
candidates/context:           81
future utility rows:       76,464
historical rows for U2:    18,432
finite Q0 chains/budget:    2,048
```

The evaluation seal, label-view manifest, evaluation table identity, complete
field, canonical candidate order, Stage-B action inputs, and historical method
table are all bound. Protected payloads were not opened in C85URP.

## Implementation

U1 is a protected utility-production subprocess. It uses the immutable
evaluation view and persisted target logits, applies the historical
bAcc/NLL/ECE and oriented-midrank formula, and atomically freezes 944
context NPZs plus a 76,464-row index. It has no selection or Q0 input.

U2 is a separate read-only subprocess. It receives only U1, frozen Stage-B
actions, and the historical method-context table. It replays selected utility,
standardized regret, top1/top5/top10, and selected regime for all 18,432 rows.
Finite Q0 endpoints integrate the already frozen 2,048 chains without
resampling. It has no label or logit input and cannot run scientific inference.

## Numerical And Publication Contract

```text
float arrays:                  little-endian float64
metric/utility replay:         max abs <= 1e-12
dtype/shape/digest/order:      exact
midrank/top-k/identity:        exact
U1 publication:               one complete staging-to-final rename
partial U1 field accepted:     false
U2 mismatch accepted:          false
runtime widening:              forbidden
```

## Authorization And Resources

The lock is unauthorized. Future execution requires a new exact direct
statement:

```text
授权 C85U
```

The authorization record is globally single-use through an external
`O_CREAT|O_EXCL` receipt and binds one exact output root. The locked resource
envelope is `cpu-high`, 48 CPU, 128 GiB RAM, zero GPU, two hours, and at most
2 GiB output.

## Readiness Counters

```text
evaluation-label rows opened:       0
target/logit NPZ payloads opened:    0
Q0 shards opened:                    0
direct C84S tables opened:           0
real utilities computed:             0
real historical rows replayed:       0
C85U authorization records:          0
C85U executions:                     0
C85E/C86 executions:                 0 / 0
```

## Validation

```text
shadow/lock tests:  19 / 19 PASS
final red team:     60 / 60 PASS
focused:            394 passed
C65:                1,067 passed, 1 skipped, 5 deselected
C23:                1,478 passed, 1 skipped, 5 deselected
full:               2,402 passed, 1 skipped, 5 deselected
accepted stderr:    empty
```

## Scientific Boundary

C84 remains `C84-D / C84-L4`. C85 theorem statuses remain T1/T3/T4/T7
`PROVED`, T2/T6 `COUNTEREXAMPLE`, and T5 `OPEN`. C85URP adds no scientific
statistic, selector, p-value, theorem test, or empirical claim.

Successful future C85U must stop at:

```text
C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED
```

It does not authorize C85E. C85EP2, C85E, C86, active acquisition, new
data/model zoos, and manuscript changes all require separate review and locks.
