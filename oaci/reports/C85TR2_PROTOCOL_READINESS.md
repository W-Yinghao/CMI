# C85TR2 Protocol Readiness

## Final Disposition

```text
C85T_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_REPAIRED_V3_LOCK_READY_FOR_PI_AUTHORIZATION
```

C85TR2 completed the additive, no-scientific-execution repair requested by the
PM. It created one prospective C85T V3 execution lock and stopped before any
authorization record, registered S0-S10 draw, canonical proof candidate, or
theorem-status transition.

## Authoritative Identity

```text
entering HEAD:
  dd75d52be4414cc893c5a2fddf0374e01e13137a

C85TR2 protocol commit:
  2e79f304202faffb857610e273ec5510a608080a

C85TR2 protocol SHA-256:
  f9a1db908f34818b7551c0d4f8de65fa7a11e71c41b8e5fe28824f042904a844

C85TR2 implementation commit:
  d489bd428f1c39a6eb399c9697b983d0b143ec80

C85T V3 lock commit:
  b1a5ba3aca002de7e302fc375298cc69c1ed82a8

C85T V3 lock SHA-256:
  3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9

runtime-bound repository objects:
  160
```

The historical V2 lock remains byte-identical:

```text
C85T_EXECUTION_LOCK_V2.json SHA-256:
  0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719

status:
  SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION
```

## Protocol Chronology

The repair protocol and timing audit were committed and pushed before any V3
implementation byte. The implementation and shadow contracts were then
committed and pushed before the V3 runtime registry and lock were materialized.

```text
protocol commit       2e79f304
  -> implementation   d489bd42
  -> V3 lock           b1a5ba3a
```

Pre-repair and post-readiness protected counts are unchanged:

```text
registered S0-S10 draws:          0
canonical proof candidates:       0
theorem-status transitions:       0
C85T V3 authorization records:    0
real project arrays:               0
active acquisition:                0
```

## Repair A: Receipt-Validated Execution Context

The operative V3 path contains no `_CAPABILITY_SENTINEL`,
`_ISSUED_CAPABILITIES`, `_issue_capability`, or generic
`consume_authorization_once` API. The only context factory accepts three paths:

```text
committed V3 lock path
committed V3 authorization-record path
exact CLI output root
```

It inseparably performs:

1. Lock sidecar, schema, status, protocol, byte, and Git-blob replay.
2. Branch `oaci`, clean worktree, `HEAD == origin/oaci`, and ancestry replay.
3. Committed authorization path, content, chronology, and protected-field replay.
4. Exact content-addressed output and normalized consumption-root replay.
5. External `O_CREAT|O_EXCL` receipt creation.
6. Receipt and containing-directory `fsync`.
7. Matching `AUTHORIZATION_CONSUMED` lifecycle event.
8. Construction of `ValidatedC85TExecutionContext` bound to the receipt inode,
   authorization file/binding hashes, lock, attempt, output root, and HEAD.

Every V3 exact, Monte Carlo, proof-candidate, and registered RNG entrypoint
revalidates the receipt and lifecycle. This is an official-result governance
control, not a claim of adversarial Python or OS security.

## Repair B: One Atomic Transaction

The authoritative lifecycle is inside the staging bundle and reaches:

```text
ATOMIC_PUBLISH_COMMIT_READY
```

before publication. Prior to the one final rename, the runtime writes and
replays all result artifacts, the semantic receipt, manifest, lifecycle, and
completion receipt; then it fsyncs every file and directory. The commit method
performs exactly:

```text
os.replace(staging_bundle, final_bundle)
return precomputed_in_memory_completion
```

There is no callback, lifecycle append, manifest replay, completion write, or
other required filesystem operation after the rename. A later recovery process
classifies a valid terminal final bundle as recovered success. A nonterminal
failure appends `FAILED` once; a terminal-without-final state emits a separate
reconciliation blocker without attempting a contradictory terminal append.

## Repair C: Primary-Exception Precedence

The primary exception is never replaced by lifecycle, cleanup, quarantine,
recovery, or reporting errors. Secondary failures are retained in a separate
list. Shadow tests cover both nonterminal failure and terminal staging failure,
including an injected lifecycle append error.

## Repair D: Result-Semantic Replay

V3 derives counts and semantic identities from persisted files. It verifies:

```text
exact keys:                         S0 through S10 exactly
S10 arithmetic:                     11/40, 0, 3/5, 13/40
S8 rational certificate:            all required LP fields
S6/S7 replicate IDs:                4,096 each
S6/S7 logical rows:                 8,192 total
S9 replicate-design rows:           8,192
S9 raw digest rows:                 4,096
S9 raw dtype/counts:                <i8 / 51 / 46
proof candidates:                   T1 through T7 exactly
formal theorem statuses:            OPEN / 7
protected counters:                 zero
```

Selected actions, indicators, regrets, paired arrays, and aggregates are
rederived from saved NPZ arrays. S9 raw digests are lowercase 64-hex and are
replayed by rerunning the exact int64 PCG64DXSM stream under the already
consumed context. Proof CSV hashes must equal file hashes, and each Markdown
statement SHA must equal the lock-bound theorem statement.

## C85T/C85V Separation

C85T V3 may freeze synthetic outputs and seven proof candidates. It cannot
claim independent proof review or transition T1-T7. All formal statuses remain
`OPEN`. A later C85V requires separate PM approval and must not rerun the Monte
Carlo benchmark.

## Validation

```text
C85TR2 direct shadow/lock tests:
  35 passed

formal focused:
  410 passed

C65 cumulative:
  1,021 passed, 1 skipped, 3 deselected

C23 cumulative:
  1,432 passed, 1 skipped, 3 deselected

full OACI:
  2,356 passed, 1 skipped, 3 deselected

accepted stderr:
  0 bytes for every suite
```

The one skip is the historical C78F test explicitly marked as already
red-teamed and finalized. The three deselections are the standing unauthorized
C79 adapter tests used by prior cumulative runs.

## Disclosed Non-Accepted Attempts

Two readiness-only invocation errors were observed and corrected before the
accepted runs:

1. A test command used the login Python 3.9 environment and failed during
   collection because historical code requires Python 3.13 `dataclass(slots)`.
   No tests or registered logic executed. All accepted runs used the exact
   locked environment.
2. The first V3 lock-builder command supplied an incorrectly expanded
   implementation commit. It failed before creating a registry or lock. The
   successful invocation used exact HEAD
   `d489bd428f1c39a6eb399c9697b983d0b143ec80`.

Neither event consumed authorization, opened a registered seed, generated a
proof candidate, or changed theorem status.

## Operational Disclosure

At final audit, `squeue` showed one pre-existing user job:

```text
897842 | name bash | /bin/bash | cpu-high | RUNNING
```

It was not submitted or modified by C85TR2, and its visible name/command do not
identify it as C84, C85, or OACI. No `sacct` claim is made.

## Authorization Boundary

The V3 lock status is:

```text
LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED
```

No current authorization record exists. The authorization text supplied before
the V3 lock is not reusable. Only a new standalone statement after PM review:

```text
授权 C85T
```

may be bound to lock commit `b1a5ba3a...`, lock SHA `3ee51a99...`, one new
authorization ID, one exact content-addressed output root, and one external
consumption path.

C85V, C85E, active acquisition, real project data, new data/model zoos, and
manuscript work remain unauthorized.
