# C85UR1 Overall Report

## Disposition

```text
C85U_PROCESS_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_TRANSACTION_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION
```

C85UR1 is complete. It repaired the U1/U2 runtime boundaries, protected-input
receipt semantics, per-stage attempt binding, and final acceptance transaction.
It did not authorize or execute C85U and performed no real protected read.

## Locked Identities

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

runtime-bound repository objects:
  54
```

The historical V1 lock remains byte-identical and is classified
`SUPERSEDED_BEFORE_AUTHORIZATION_OR_REAL_PROTECTED_ACCESS`.

## Runtime Isolation

U1 now imports a dedicated registry that contains only the complete field,
trial/candidate identities, evaluation seal/view/table identity, target
artifacts and sidecars, and context descriptors. It neither defines nor opens
Stage-B selections, Q0 inputs, scientific results, method decisions, or
inference objects. A dynamic open trap validates that boundary.

U2 has no hard-coded project paths and no unguarded real entrypoint. It requires
the same validated authorization/lock/attempt context, completed U1 V2 handoff,
U1 manifest, lifecycle event, and one fresh `O_EXCL` U2 receipt before resolving
any Stage-B or historical-result input.

## Protected Replay And Stage Objects

The V2 protected-input receipt binds authorization and lock identities, attempt
and root, evaluation-view identities, exactly 1,944 target artifacts and
sidecars, registry digests, and 48,018,748,054 target bytes. U1 parses and
semantically revalidates these fields before protected payload access.

Versioned U1 and U2 manifests/handoffs bind the same attempt. U2 retains exact
coverage of 18,432 historical endpoints and 8,749,056 finite Q0 action records
without resampling. Both child roots are provisional and explicitly not
accepted for C85E.

## Atomic Acceptance

The final acceptance bundle contains execution result, artifact manifest,
completion receipt, lifecycle, authorization receipt, protected-input replay
receipt, and U1/U2 identities. All validation, writes, terminal lifecycle
recording, and fsyncs occur in staging. Publication is one final `os.replace`,
with no required operation afterward.

Only the final bundle may contain:

```text
C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED
```

A valid post-rename bundle is recovered as success. U1 success followed by U2
failure leaves U1 provisional and creates no acceptance bundle.

## Scope And Resources

```text
contexts:                         944
candidates/context:                81
future utility rows:           76,464
target artifacts/sidecars:    1,944 / 1,944
target artifact bytes:    48,018,748,054
historical endpoint rows:       18,432
finite Q0 action records:    8,749,056

partition / CPU / RAM / GPU:
  cpu-high / 48 / 128 GiB / 0

wall / U1 output maximum:
  2 hours / 2 GiB
```

## Readiness Counters

```text
evaluation-label rows opened:       0
target/logit payloads opened:        0
target-sidecar payloads opened:      0
Q0 shards opened:                    0
direct C84S tables opened:           0
real utilities computed:             0
real endpoints replayed:             0
C85U authorization/execution:        0 / 0
C85E/C86 execution:                  0 / 0
training/forward/GPU:                 0 / 0 / 0
```

## Validation

```text
post-lock C85UR1 tests:  21 / 21 PASS
final red team:          72 / 72 PASS
focused:                 396 passed
C65:                     1,088 passed, 1 skipped, 5 deselected
C23:                     1,499 passed, 1 skipped, 5 deselected
full:                    2,423 passed, 1 skipped, 5 deselected
accepted stderr:         empty
```

## Boundary

C84 remains `C84-D / C84-L4`. C85 theorem statuses remain T1/T3/T4/T7
`PROVED`, T2/T6 `COUNTEREXAMPLE`, and T5 `OPEN`.

The V2 lock is unauthorized. Future execution requires a new standalone:

```text
授权 C85U
```

C85EP2 must independently replay U1, U2, and the final acceptance bundle before
any C85E lock. C85E, C86, active acquisition, new data/model zoos, and
manuscript work remain unauthorized.
