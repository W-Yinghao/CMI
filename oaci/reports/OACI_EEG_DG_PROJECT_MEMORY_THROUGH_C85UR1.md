# OACI EEG-DG Project Memory Through C85UR1

## Current State

```text
milestone:
  C85UR1

gate:
  C85U_PROCESS_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_TRANSACTION_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION

C85U V2 execution lock:
  present and unauthorized

C85U real protected execution:
  0

C85E / C86:
  not authorized
```

C85UR1 supersedes the non-operative V1 execution lock without changing the
historical C85U protocol or any C84/C85 scientific object. The V1 lock remains
immutable and was never authorized or used for protected access.

## C85UR1 Chronology

```text
starting HEAD:
  c110f15b054ae7a9aee7cd5cea4d7f78f6891a95

C85UR1 protocol commit:
  1cc5531c02571fa95c33b3450c6b7d6ff53a8ebf

C85UR1 protocol SHA-256:
  aa657133d35602187a5c5e11a9632a44c26a78fb63a4e65197172fb59377061d

C85UR1 implementation commit:
  5917cf54d33bbd5de906428bea5aaee22f45aabb

C85U V2 execution-lock commit:
  672670d05e9d7adfbe12673d4a64bfd499413162

C85U V2 execution-lock SHA-256:
  77382c16a593f7c2bdeb4dcacdfa21df11dcfd59982e9bfb982d6b88f5f04d1d
```

The protocol was committed and pushed before implementation. The implementation
was committed and pushed before the 54-object V2 lock was constructed.

## Repaired Execution Boundary

U1 now uses a dedicated runtime registry with no Stage-B selection, Q0,
scientific-result, method-decision, or inference paths. Before evaluation-label
or target-NPZ access it must validate the lock, consumed authorization, the
semantic V2 protected-input receipt, lifecycle identity, and one fresh U1
stage receipt.

U2 now requires the same authorization/lock/attempt context, complete U1 V2
handoff and manifest, `STAGE_U1_COMPLETED`, and one fresh U2 stage receipt. It
contains no hard-coded protected paths and cannot open a Stage-B/Q0/result
object before those guards pass.

U1 and U2 remain provisional. C85E acceptance exists only in a final staging
bundle containing their identities, lifecycle, receipts, and acceptance
manifest. All required work occurs before one final rename, with no required
operation afterward. A valid post-rename bundle is recovered as success; U2
failure leaves no acceptance bundle.

## Readiness Evidence

```text
post-lock C85UR1 tests:  21 passed
focused:                 396 passed
C65:                     1,088 passed, 1 skipped, 5 deselected
C23:                     1,499 passed, 1 skipped, 5 deselected
full:                    2,423 passed, 1 skipped, 5 deselected
red team:                72 / 72 PASS
accepted stderr:         empty
```

No evaluation-label row, target/logit payload, target sidecar payload, Q0 shard,
or direct C84S result table was opened. No real utility or historical endpoint
was computed. No C85U authorization record exists.

Future execution requires a new standalone `授权 C85U` and must stop at
`C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED`. C85EP2
must independently replay the accepted result before any C85E lock.

## Prior State Through C85URP

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
