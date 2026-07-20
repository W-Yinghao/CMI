# OACI EEG-DG Project Memory Through C85URP

## Current State

```text
milestone:
  C85URP

gate:
  C85U_HELD_EVALUATION_CANDIDATE_UTILITY_RECONSTRUCTION_LOCKED_READY_FOR_PI_AUTHORIZATION

C85U execution lock:
  present and unauthorized

C85U real execution:
  0

C85E / C86:
  not authorized
```

C85EP previously stopped correctly because no complete frozen 944 x 81
held-evaluation utility field existed. C85URP prospectively defines a new
protected C85U production milestone; it does not rewrite that historical
blocker.

## C85URP Chronology

```text
C85EP blocker HEAD:
  c470084d94910bfe7290a8565db30d058d5b3de6

C85U protocol commit:
  ebe158c9e929f67423a9ebdc3cea7c6ea5c16c9a

C85U protocol SHA-256:
  c9ed7081cf8cb1a6c8a05181d1660da2015b4e1716a05c8916f7fe5b09efc160

C85U implementation commit:
  df100e2e77c5749030e2931bb7752973258823bd

C85U execution-lock commit:
  3b4fa48ee2d4f75ff8ba2191dc7d8593237dc82f

C85U execution-lock SHA-256:
  923c6bee2171f0bedcc3f883058759d368bdb49eb272cbbfa80974e98b632fe1
```

The protocol was committed before metadata discovery and implementation. The
implementation was committed before the execution lock. All are pushed on
`origin/oaci`.

## Frozen C84/C85 State

```text
C84 primary:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

C84 label frontier:
  C84-L4

C85 theorem statuses:
  T1 PROVED
  T2 COUNTEREXAMPLE
  T3 PROVED
  T4 PROVED
  T5 OPEN
  T6 COUNTEREXAMPLE
  T7 PROVED
```

C85URP changes no scientific result or theorem status.

## Bound Input Field

```text
C84F field descriptors:       1,944
C84F target artifacts:        1,944
target-artifact bytes:         48,018,748,054
C84 contexts:                    944
candidates/context:               81
future candidate utilities:    76,464
historical method rows:        18,432
```

The C84F complete-field manifest, C84S V5 lock, Stage-A evaluation seal/view,
selection freeze, scientific result, result manifest, operative unit registry,
target paths/hashes, and canonical context/candidate order are bound.

The evaluation label table is 394,109 bytes with 4,848 rows. Its identity is
bound from the immutable view manifest, but its rows were not opened. The
1,944 target NPZ payloads, Q0 shards, candidate ranks, fixed actions, and
historical method table were also not opened in C85URP.

## Future C85U Architecture

### U1

U1 is an authorization-gated subprocess. It receives the evaluation view,
persisted target logits, and field descriptors only. It computes the historical
bAcc/NLL/ECE midrank composite for all 81 candidates in each context and
freezes:

```text
944 context NPZ artifacts
76,464-row candidate_utility_index.csv
C85U_CANDIDATE_UTILITY_MANIFEST.json
```

It has no construction-label, Stage-B, Q0, selector, or inference input.

### U2

U2 is a separate read-only subprocess. It receives U1, immutable Stage-B
actions, and `method_context_decisions.csv`. It replays 18,432 selected utility,
regret, top-k, and regime rows. Finite Q0 uses the existing 2,048-chain action
records with no resampling. U2 has no label/logit input and cannot call Q1/Q2,
max-T, LOTO, frontier, or taxonomy.

## Persistence And Failure Contract

```text
float dtype:
  little-endian float64

metric/utility replay:
  max abs <= 1e-12

identity/dtype/shape/digest/order/midrank:
  exact

partial U1 publication:
  forbidden

U2 mismatch:
  U1 retained but not accepted for C85E

automatic retry:
  forbidden
```

## Authorization And Resources

Future authorization requires a fresh standalone:

```text
授权 C85U
```

The record uses `c85u_direct_pi_authorization_record_v1`, binds one exact
content-addressed root, and is consumed once through an external exclusive
receipt. C85E, C86, active acquisition, new data/model zoos, and manuscript
work remain false.

Locked execution resources:

```text
cpu-high / 48 CPU / 128 GiB / 0 GPU / 2 hours / <=2 GiB output
```

## Readiness Access Counters

```text
evaluation-label rows:             0
target/logit payloads:              0
target-sidecar payloads:            0
Q0 shards:                          0
direct C84S tables:                 0
real utility rows:                  0
real historical replay rows:        0
C85U authorization/execution:       0 / 0
```

## Validation

```text
C85URP tests: 19 passed
red team:     60 / 60 PASS
focused:      394 passed
C65:          1,067 passed, 1 skipped, 5 deselected
C23:          1,478 passed, 1 skipped, 5 deselected
full:         2,402 passed, 1 skipped, 5 deselected
stderr:       empty for every accepted run
```

## Authoritative C85URP Reports

```text
oaci/reports/C85URP_OVERALL_REPORT.md
oaci/reports/C85URP_OVERALL_REPORT.json
oaci/reports/C85URP_OVERALL_REPORT.sha256
oaci/reports/C85URP_PROTOCOL_READINESS.md
oaci/reports/C85URP_FINAL_REPORT_RED_TEAM.md
oaci/reports/C85URP_REGRESSION_VERIFICATION.md
oaci/reports/C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.json
oaci/reports/C85U_EXECUTION_LOCK.json
oaci/reports/c85urp_tables/
```

## Continuation Boundary

C85U is ready but unauthorized. Successful future C85U stops at:

```text
C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED
```

Then C85EP2 must independently replay the complete U1 field and U2 historical
decision audit before creating any C85E execution lock. C85U completion alone
does not authorize C85E.
