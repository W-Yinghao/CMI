# C84S V4 Authorized Execution Blocker

## Disposition

The PI directly authorized C84S against the unique V4 analysis lock. Job
`898192` consumed that authorization, replayed the immutable Stage-A label
views, and entered Stage B with the evaluation descriptor still sealed.

Stage B stopped before the first complete selection context because the
registered Lee2019_MI secondary Q0 budget `B=32` requires 32 construction
labels per class, while every Lee target has exactly 25 construction labels per
class under the locked split.

```text
V4 authorization record commit:
  d699919f8d3b96a2f2e42e94a09df076a9af52d4

V4 analysis-lock SHA-256:
  582e5074b4b17d62ff1e5fbfd992f037dd3082b7763b22d707630aa19db81c3d

authorization consumption SHA-256:
  6dfc058e67ea8fa1ea8ddc0c1d398a4b468c4213a42455b5f864ced800fb0866

job:
  898192

runtime:
  2026-07-16T10:34:34Z to 2026-07-16T10:44:20Z

final lifecycle status:
  FAILED

operative gate:
  C84S_REAL_EXECUTION_SELECTION_AGGREGATION_RESOURCE_OR_PROVENANCE_RECONCILIATION_REQUIRED
```

No C84S scientific result exists.

## Passed Replays

Before authorization consumption, the runtime passed:

```text
bound repository and protocol replay: PASS
external field artifacts:             7,776 / 7,776
external bytes rehashed:               48,072,941,176
environment and loader identities:     PASS
HEAD == origin/oaci:                   PASS
fresh output root:                     PASS
```

After consumption, immutable Stage-A replay passed without a loader call or a
target-label reload:

```text
historical Stage-A files: 11 / 11
label-loader calls:        0
label rows reloaded:       0
selector scores:           0
scientific statistics:     0
```

## Primary Blocker

The Q0 budget is defined historically as labels **per class**. The construction
view contains:

| Dataset | Targets | Minimum labels/class | B=8 | B=16 | B=32 |
|---|---:|---:|---:|---:|---:|
| Lee2019_MI | 22 | 25 | 22/22 | 22/22 | 0/22 |
| Cho2017 | 20 | 50 | 20/20 | 20/20 | 20/20 |
| PhysionetMI | 76 | 9 | 76/76 | not registered | not registered |

Thus the three-dataset primary grid `[1,2,4,8,FULL]` is feasible and unchanged.
Lee `B=16` and Cho `B=16/32` are feasible secondary analyses. Lee `B=32` is
physically unavailable for every target and cannot be sampled without changing
the locked without-replacement policy.

The exact primary exception was:

```text
C84SContractError: Q0 budget 32 is infeasible
```

## Secondary Cleanup Error

While propagating the Q0 exception, the atomic publisher attempted to remove
the Stage-B staging directory before all stream handles were closed. The NFS
cleanup raised:

```text
OSError: [Errno 39] Directory not empty
```

This secondary cleanup exception masked the primary error in the Stage-B
attempt ledger. The full Stage-B stderr preserves both tracebacks. No final
selection-freeze directory was published. The empty residual hidden staging
directory remains in the failed root as evidence and must not be reused.

## Protected Counters

```text
construction-label access: 1
evaluation-label access:   0
selector-score contexts:   0
scientific result rows:    0
training / forward / GPU:  0 / 0 / 0
same-label oracle:         0
```

The evaluation descriptor remained sealed throughout the failed attempt.

## Required Additive Repair

Before another C84S execution:

1. preserve V4, job `898192`, its consumed authorization, and its failed root;
2. mark Lee `B=32` as input-unavailable while retaining Cho `B=32` and all
   feasible registered budgets;
3. update exact Q0 and method-context arithmetic without changing the primary
   grid, methods, thresholds, inference, or taxonomy;
4. ensure stream handles close before cleanup and that cleanup errors never
   replace the primary scientific or implementation exception;
5. run a full production-path synthetic calibration with Lee's exact 25-label
   construction capacity;
6. create a replacement analysis lock and obtain a fresh direct PI
   authorization.

The consumed V4 authorization cannot migrate to the replacement lock.
