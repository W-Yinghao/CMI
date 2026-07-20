# C80R - Additive Repair and Existing-Field Adapter Readiness

## Final gate

```text
C80_REPAIRED_PROTOCOL_AND_REAL_ADAPTER_LOCKED_READY_FOR_PI_REAUTHORIZATION
```

C80R is complete as a protocol, implementation, relock, and readiness
milestone. C80E scientific execution is not authorized. No real label-budget
statistic or evaluation-label value was read.

## Operative objects

```text
safe-stop evidence commit:  6c18fd4bdd794b32ceffd0c9cd3da8fbc487ab69
repair protocol commit:      e88a24484590636f87d0f22798401a762875046a
repair protocol SHA-256:     2d72eb5119056a6520fd33fc0ac14ee6270bfd573b59c36b74be6aa3dc25fe39
final adapter commit:        e5cb41a5cd389674e3ec201d5c5f68a361c2fed3
final adapter SHA-256:       7e5ac0ba829bf5f233ed469f6fb8f6da4054d0bf4d024a0736a45e3674f1b56c
analysis lock commit:        f19acd8775f9b0ddf60401739741bec0019d021c
analysis lock SHA-256:       e18f2b5f1d79b6fcd96207339c5842e30b7aecb5bc22b8939a475487068b1b82
field/view manifest digest:  6180275dcef26bdda4ae4b291d1ef6dc83434462ecacee0350fa94ae9c6a7fef
```

The historical protocol `f5d83b3`, hash `c629...dd85`, and analysis lock
`972f47c` remain preserved but are superseded for execution. Their prior C80E
authorization reached preflight only and is not reusable.

## Closed blockers

C80R closes the three accepted pre-outcome blockers additively:

1. The machine protocol now contains a mutually exclusive, exhaustive C80-E,
   C80-D, C80-B, C80-C, C80-A priority table.
2. `near-FULL` is the ordinal set `{32,FULL}`. `FULL` remains cell-specific,
   is not a universal 61 labels/class, and is not interpolated with 32.
3. The authorization guard uses canonical `lock.protocol.sha256`, and the
   replacement lock binds a fail-closed real-data adapter plus exact field/view
   manifests and all report schemas.

The final taxonomy applies in this order:

```text
1 C80-E: a primary execution/view/dependence/protocol/provenance blocker
2 C80-D: either seed-specific B* is absent
3 C80-B: both B* exist but registered cross-seed stability fails
4 C80-C: stability passes and both B* are in {32,FULL}
5 C80-A: stability passes and C80-C is false
```

## Locked execution boundary

The adapter is bound to the exact seven-point grid
`[1,2,4,8,16,32,FULL]`, Q0 nested stratified uniform sampling, 2,048 PCG64
chains, the exact C79 P1 selector, target-cluster simultaneous inference, and
all five unconditional registry paths P1/P2/S1/S2/S3. Registry completeness is
80/80 with zero blank cells.

Runtime order is fail-closed:

```text
protocol/lock/manifest/new-authorization replay
-> construction-only nested-Q0 selection
-> content-addressed selection freeze
-> selection hash replay
-> evaluation-view opening
-> unconditional P1/P2/S1/S2/S3 execution
-> machine-readable result freeze
```

Primary targets remain `[1,2,3,5,6,7,8,9]`. Target 4 is excluded. The
same-label oracle is unreachable. Trial IDs and row order are alignment,
partition, and dependence keys only. C80R read compact manifests and
synthetic/schema fixtures; it did not open external NPZ payloads.

## Protected state

```text
real budget statistics:        0
evaluation-label value reads:  0
same-label oracle accesses:    0
target4 primary rows:          0
training/forward/re-inference: 0
GPU jobs:                      0
new C80E authorization:        absent
```

The pre-execution red team passed 40/40 checks with zero blockers. No
scientific registry entry, estimand, selector, budget, threshold, RNG stream,
Monte Carlo count, dependence rule, or materiality rule changed during the
repair.

## Regression

All final suites ran on exact clean commit
`93d2099f14b8739089e640c0e6078f02ed5cc435`:

```text
focused:    53 passed
C65-C80R:  368 passed, 1 conditional skip, 3 explained deselections
C23-C80R:  775 passed, 1 conditional skip, 3 explained deselections
full OACI: 1703 passed, 1 conditional skip, 3 explained deselections
```

Failures were zero and every stderr file was empty. The skip is the finalized
C78F guard. The deselections are historical C79P preauthorization-state tests;
no C80R path was skipped or deselected. Earlier log-placement and test-glob
issues are preserved as closed non-scientific repairs.

## Reauthorization boundary

Future C80E execution requires a new direct PI authorization binding the
operative protocol commit/hash, analysis lock commit/hash, and manifest digest
listed above. Until that record exists, `run-real` fails before any external
array load.

Any future authorization may cover only the frozen seed3/seed4 existing-field
C80 analysis. It does not authorize training, forward/re-inference, GPU,
target4 primary use, same-label-oracle work, BNCI2014_004, seed5, active
acquisition, a new feature/kernel/model search, C81, or manuscript drafting.
