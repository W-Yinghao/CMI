# C84SR3 Overall Report

## Final Disposition

```text
C84S_SECONDARY_Q0_AVAILABILITY_AND_ATOMIC_FAILURE_REPAIRED_V5_LOCK_READY_FOR_FRESH_PI_AUTHORIZATION
```

C84SR3 completed the additive Q0 secondary-budget availability and atomic
failure repair and created a fresh C84S V5 analysis execution lock. It did not
execute real C84S science. A new direct `授权 C84S` statement is required against
the V5 lock.

## Chronology And Identities

```text
C84SR3 protocol commit:
  91f984503fa84b53fae32948d0cf49e7ede12b8f

C84SR3 protocol SHA-256:
  5c783db9113697b2c710af4c1f1bafd66a3096be7a1b5cbac8aa03ca2a9c3080

production implementation commit:
  15dec7d02226738a4147df0034767626c375fdf6

lock-bound readiness implementation commit:
  815d0ccd3f2ef245ea66c734165905d3a08ac105

V5 analysis-lock commit:
  2d03eb05e0cec352d08cdb6f48170be56876e77b

V5 analysis-lock SHA-256:
  030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846

accepted regression commit:
  4cbe49d68b280d90c3b49ec82cbbbf9e8df95ed9
```

The repair protocol preceded implementation. The first readiness command used
an incorrect expanded implementation hash and stopped before writing a lock.
That pre-lock operator error is retained in the failure ledger. The corrected
readiness run bound the exact current implementation commit.

## Preserved V4 Attempt

Authorized job `898192` remains an immutable failed attempt:

```text
authorization consumed:       yes
construction-label access:    1
evaluation-label access:      0
selector contexts completed:  0
scientific result rows:       0
training / forward / GPU:     0 / 0 / 0
same-label oracle:            0
selection freeze published:  no
evaluation descriptor sealed: yes
```

Its primary failure was `Q0 budget 32 is infeasible` for Lee2019_MI. During
failure cleanup, an NFS `errno 39` then masked that primary exception in the
attempt ledger. The V4 authorization, consumption receipt, failed root, partial
staging directory, logs and blocker reports remain preserved. They are not
reusable by V5.

## Construction Availability

The construction-only audit established that Q0 budgets are labels per class:

| Dataset | Targets | Construction labels/class | Operative finite budgets |
|---|---:|---:|---|
| Lee2019_MI | 22 | 25 exactly | 1, 2, 4, 8, 16 |
| Cho2017 | 20 | 50 exactly | 1, 2, 4, 8, 16, 32 |
| PhysionetMI | 76 | 9 to 15 | 1, 2, 4, 8 |

The primary grid `[1,2,4,8,FULL]` is unchanged for all datasets. Lee B16
remains secondary and feasible. Lee B32 remains a historically registered
secondary item but is physically input-unavailable for all 22 targets; it has
no selection row and no result row. Cho B16/B32 remain operative secondary
curves. There is no sampling with replacement, target-specific substitution,
budget redefinition, or replacement by `FULL`.

## Repaired Arithmetic

```text
contexts:                         944
candidates/context:               81
candidate-score rows:        535,248
candidate-rank rows:         535,248
fixed-default rows:            4,720
Q0 shards:                       944
Q0 finite/FULL records:    8,750,000
Q0 sample-digest rows:     1,093,750
Stage-B Q0 regime rows:       15,648
Stage-B Q0 coverage rows:      5,216
Stage-C method-context rows:  18,432
Stage-C Q0 regime rows:       12,816
Stage-C Q0 MC rows:            4,272
```

Method-context rows are 3,520 for Lee, 3,360 for Cho and 11,552 for
Physionet. Q0 still uses 2,048 paired chains and one deterministic `FULL`
record per context. Sampling-plan identities are checked across panel, seed and
level repetitions for each dataset/target.

## Atomic Failure Repair

Stage B now closes all streaming CSV handles before staging cleanup. Cleanup is
bounded and cannot replace the primary exception, including under the locked
Python 3.9 runtime. If NFS cleanup still cannot remove a hidden staging
directory, the residual path is preserved and attached to the original error.
No partial Stage-B directory is publishable.

The V5 Stage-A replay has its own authorization-receipt entrypoint and accepts
only a `C84S_V5_authorization_consumed` receipt. It replays the immutable
historical Stage-A label views without invoking a loader. The evaluation
descriptor remains sealed until a complete Stage-B V3 selection freeze passes.

## Full-Scale Synthetic Calibration

The exact production entrypoints passed an end-to-end synthetic run with:

```text
contexts:                    944 / 944
Q0 chains:                 2,048 / 2,048
Q0 records:            8,750,000 / 8,750,000
method-context rows:      18,432 / 18,432
real field/label access:       0 / 0
```

All C84-A/B/C/D/E and C84-L1/L2/L3/L4 branches matched their truth tables.
Lee B32 was absent and Cho B32 was present. The synthetic summary SHA-256 is
`b041fa8f2fd7ffba911adce6995d5171995e2bf946dba1913e8bf4443d4fb0c8`.

## Frozen Field Replay

Readiness replayed 1,944 field descriptors and byte-hashed 7,776 external
selection artifacts totaling 48,072,941,176 bytes. The complete C84F field,
method registry, immutable Stage-A views, historical failed attempts and all
V5 runtime implementation bytes are bound by the execution lock.

## Regression Verification

Independent `cpu-high` Slurm jobs, monitored with `squeue`, produced:

```text
focused:  367 passed
C65:      853 passed, 1 skipped, 3 deselected
C23:    1,264 passed, 1 skipped, 3 deselected
full:   2,188 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. `sacct` was not used. The skip and three
deselections are the established C78F finalized test and C79 historical
authorization-state tests. Initial default-environment and historical-lifecycle
test failures are retained in `regression_attempt_ledger.csv`.

## Scientific Boundary

C84SR3 is readiness only. During the repair:

```text
new target-label reload:      0
evaluation-label access:      0
real selector scores:         0
real scientific statistics:   0
training / forward / GPU:      0 / 0 / 0
same-label oracle:             0
C85 authorization:            false
```

No C84-A-E or C84-L1-L4 scientific result exists. C84SR3 does not establish
external validity, selector performance, Q1/Q2, a label frontier or a level
effect.

## Next Valid Action

The V5 lock status is:

```text
LOCKED_READY_FOR_FRESH_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED
```

The shortest valid next authorization is:

```text
授权 C84S
```

That future statement must be bound to the V5 lock SHA above. The consumed V4
authorization does not migrate, and C84S authorization does not authorize C85.
