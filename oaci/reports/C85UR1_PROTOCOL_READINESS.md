# C85UR1 Process-Isolation And Acceptance-Transaction Readiness

## Final Gate

```text
C85U_PROCESS_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_TRANSACTION_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION
```

C85UR1 repaired the C85U execution boundary and created an unauthorized V2
lock. It did not execute C85U, create an authorization record, open an
evaluation-label row, open a target NPZ payload, open a Q0 shard or direct
C84S result table, or compute a real candidate utility.

The V1 lock remains byte-identical and is classified as:

```text
SUPERSEDED_BEFORE_AUTHORIZATION_OR_REAL_PROTECTED_ACCESS
```

## Authoritative Chronology

```text
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

The additive protocol was committed and pushed before any V2 implementation
change. The implementation was then committed and pushed before the lock was
built. The V2 lock binds 54 repository objects by size, SHA-256, and Git blob.

## Preserved V1 Evidence

```text
C85U V1 protocol SHA-256:
  c9ed7081cf8cb1a6c8a05181d1660da2015b4e1716a05c8916f7fe5b09efc160

C85U V1 lock SHA-256:
  923c6bee2171f0bedcc3f883058759d368bdb49eb272cbbfa80974e98b632fe1

V1 authorization records:
  0

V1 real protected access:
  0
```

No V1 file was edited. V1 utility formulas, schemas, tests, reports, and
regression evidence remain historical inputs. The C85EP availability blocker
also remains historically correct; C85UR1 does not retroactively make C85EP
ready.

## Exact Scope

```text
target contexts:                    944
candidates per context:              81
future candidate-utility rows:    76,464
target artifacts:                  1,944
target-artifact bytes:    48,018,748,054
historical method-context rows:    18,432
finite Q0 action records:       8,749,056
Q0 chains per finite budget:        2,048
```

The historical bAcc/NLL/ECE oriented-midrank utility formula, first-index
argmax, standardized-regret convention, target/candidate universe, frozen
labels, Q0 actions, and historical decision rows are unchanged.

## U1 Runtime Isolation

The V2 U1 entrypoint imports `c85u_u1_registry_v2`, not the broad V1 readiness
registry. The U1 registry defines only:

```text
C84F complete-field and target-trial identities;
operative candidate order;
Stage-A evaluation seal/view/table identity;
1,944 target artifact and sidecar identities;
944 context descriptors.
```

It does not define or import a Stage-B selection manifest, candidate score or
rank path, fixed selection, Q0 index/shard, scientific-result JSON, result
manifest, method-context table, Q1/Q2, LOTO, frontier, or taxonomy path.

A dynamic open trap ran the complete U1 metadata builder and observed no
Stage-B or scientific-result open. Real U1 must validate the committed V2
lock, external authorization receipt, protected-input replay receipt,
lifecycle, and a fresh U1 `O_EXCL` stage receipt before it can open the
evaluation table or a target NPZ.

The locked zoo-loading order opens each of 1,944 target artifacts exactly once
across 24 candidate zoos. It enforces 944 contexts, 76,464 rows, exact access
counters, and a 2 GiB U1 artifact/index envelope.

## Protected Replay Receipt V2

The protected replay handoff is no longer accepted by file hash alone. U1
parses schema `c85u_protected_input_replay_receipt_v2` and revalidates:

```text
authorization file SHA and normalized binding SHA;
authorization ID;
execution-lock SHA and commit;
attempt ID and exact output root;
evaluation-table SHA and 4,848-row identity;
evaluation-view manifest SHA;
1,944 target-artifact rows and 48,018,748,054 bytes;
target-artifact registry SHA;
1,944 target-sidecar rows and registry SHA;
replay completion timestamp.
```

A validly rehashed receipt with the wrong schema fails. Receipts from another
authorization, attempt, or output root fail before protected payload access.

## U2 Runtime Isolation

The V2 real U2 command requires all four arguments:

```text
--execution-context
--u1-handoff
--utility-root
--output-root
```

There is no real V2 path accepting only utility/output roots. U2 first replays
the same authorization, lock, attempt, parent root, U1 stage receipt, complete
U1 V2 manifest/handoff, protected-replay SHA, and `STAGE_U1_COMPLETED`
lifecycle event. It then consumes one fresh U2 `O_EXCL` receipt. Only after
these checks does it resolve Stage-B/result paths from the validated lock.

The U2 registry module contains no absolute project path and no evaluation,
target-artifact, or logit location. U2 receives only the accepted U1 utility
field, frozen ranks/fixed actions/Q0 action shards, and historical
`method_context_decisions.csv`. It replays 18,432 endpoints and 8,749,056
finite Q0 action records without resampling. Its result remains provisional
and explicitly sets `accepted_for_C85E: false`.

## Versioned Stage Objects

U1 freezes atomically:

```text
C85U_CANDIDATE_UTILITY_MANIFEST_V2.json
C85U_CANDIDATE_UTILITY_MANIFEST_V2.sha256
C85U_STAGE_U1_HANDOFF.json
C85U_STAGE_U1_HANDOFF.sha256
```

The V2 objects bind lock, authorization, attempt, protected replay, U1 stage
receipt, actual allowed/forbidden access counters, target bytes, and complete
utility identities. V1-compatible context artifacts remain available for the
historical replay code but cannot assert C85E acceptance.

U2 freezes atomically:

```text
C85U_HISTORICAL_DECISION_REPLAY_V2.json
C85U_HISTORICAL_DECISION_REPLAY_V2.sha256
C85U_STAGE_U2_HANDOFF.json
C85U_STAGE_U2_HANDOFF.sha256
```

These bind the same attempt, U1 identities, U2 stage receipt, Stage-B/result
hashes, exact coverage, maximum endpoint errors, and zero label/logit/inference
access.

## Atomic Final Acceptance

U1 and U2 are provisional child roots. C85E acceptance exists only after one
final transaction freezes:

```text
C85U_EXECUTION_RESULT.json
C85U_RESULT_ARTIFACT_MANIFEST.json
C85U_COMPLETION_RECEIPT.json
C85U_LIFECYCLE.jsonl
authorization_consumed.json
protected-input replay receipt copy
U1 acceptance identity
U2 acceptance identity
```

All semantic replay, manifest replay, completion writing, terminal lifecycle
writing, file fsync, and directory fsync occur in staging. Publication uses one
final `os.replace(staging_acceptance_bundle, final_acceptance_bundle)`. The
commit method has no function call after that rename.

The success gate appears only inside the final acceptance bundle. A valid
post-rename bundle is success or recovered success; it never receives a
contradictory `FAILED`. U1 may remain provisional after U2 failure, but no
final acceptance bundle then exists.

## Failure Precedence

The primary exception is retained if lifecycle append, cleanup, quarantine,
reconciliation reporting, or failure-receipt writing also fails. `FAILED` is
appended only to a nonterminal lifecycle. Terminal staging without the final
rename produces a separate reconciliation blocker. No consumed authorization
is retried automatically.

Preserved readiness attempts:

```text
initial shadow run:
  12 passed / 4 failed

repaired shadow run:
  16 passed

first lock build:
  rejected an incorrect manually expanded implementation SHA;
  wrote 0 lock objects

accepted lock build:
  exact implementation commit;
  3 lock objects written
```

All failures occurred before authorization and before real protected access.

## Authorization And Resources

The V2 lock remains unauthorized. A future standalone statement is required:

```text
授权 C85U
```

The future record schema is `c85u_direct_pi_authorization_record_v2`. It binds
one exact content-addressed output root and one external authorization
consumption path. C85E, C86, active acquisition, real data beyond the scoped
reconstruction, new data/model zoos, and manuscript fields remain false.

Locked resources:

```text
partition:       cpu-high
CPU:             48
RAM:             128 GiB
GPU:             0
wall:            2 hours
U1 output max:   2 GiB
```

## Readiness Access Counters

```text
real evaluation-label rows opened:       0
real target/logit payloads opened:        0
real target-sidecar payloads opened:      0
real Q0 shards opened:                    0
real direct C84S result tables opened:    0
real candidate utilities computed:        0
real historical endpoints replayed:       0
C85U authorization records created:       0
C85U executions:                          0
C85E / C86 executions:                    0 / 0
training / forward / GPU:                 0 / 0 / 0
```

## Immutable Results

```text
C84 primary:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

C84 frontier:
  C84-L4

C85 theorem statuses:
  T1/T3/T4/T7 PROVED
  T2/T6 COUNTEREXAMPLE
  T5 OPEN
```

C85UR1 changes none of these objects.

## Continuation Boundary

C85U V2 is ready for PM review and fresh authorization, but it is not
authorized. A successful future execution must stop at:

```text
C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED
```

C85EP2 must independently replay U1, U2, and the final acceptance bundle before
creating any C85E lock. C85E, C86, active acquisition, new data/model zoos, and
manuscript work remain unauthorized.
