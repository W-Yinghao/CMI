# C85TR2 Final Report Red Team

## Verdict

```text
checks: 70
PASS:   70
FAIL:    0
```

The red team finds the additive C85TR2 repair ready for PM review at:

```text
C85T_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_REPAIRED_V3_LOCK_READY_FOR_PI_AUTHORIZATION
```

This verdict is readiness-only. It does not authorize C85T, open a registered
S0-S10 stream, create a proof candidate, transition a theorem status, or
authorize C85V/C85E.

## Evidence Base

The audit replays the following committed identities:

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

V3 runtime registry SHA-256:
  a4cbdccbd872581e8b2c3ee602850e426d992c25f97591a9a474cce2754e0c55

runtime-bound repository objects:
  160
```

The historical V2 lock remains byte-identical at SHA-256
`0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719`
and is marked `SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION`.

## Red-Team Matrix

| Category | Checks | Passed | Evidence |
|---|---:|---:|---|
| chronology and historical preservation | 8 | 8 | protocol timing audit, supersession ledger, Git commits |
| authorization and validated context | 12 | 12 | context contract, adversarial tests, receipt replay tests |
| atomic transaction and recovery | 12 | 12 | state machine, recovery truth table, transaction tests |
| result-semantic replay | 16 | 16 | semantic replay table and corruption-injection tests |
| C85T/C85V proof separation | 6 | 6 | V3 lock, proof disposition checks, OPEN-status tests |
| scientific and execution isolation | 6 | 6 | static audits, zero counters, artifact-absence scan |
| formal regressions | 4 | 4 | focused, C65, C23, and full OACI logs |
| Git and operational hygiene | 6 | 6 | clean/pushed lock commit, file-size scan, `squeue` audit |
| **Total** | **70** | **70** | **no unresolved blocker** |

## Chronology And Preservation

All eight checks pass:

1. The repair protocol was committed before implementation.
2. The implementation was committed before the V3 lock.
3. The lock binds the exact implementation commit.
4. The historical V2 lock and sidecar were not edited.
5. V2 is explicitly superseded before authorization or registered execution.
6. Registered S0-S10 draws before and during C85TR2 are zero.
7. Canonical proof candidates and theorem transitions remain zero.
8. The earlier `授权 C85T` statement is not represented as V3 authorization.

## Authorization And Context Adversarial Audit

All twelve checks pass:

1. The V3 path has no operative `_CAPABILITY_SENTINEL`.
2. It has no operative `_ISSUED_CAPABILITIES` registry.
3. It has no operative `_issue_capability` helper.
4. It has no generic arbitrary-mapping consumption API.
5. The factory requires committed lock and authorization paths plus the CLI root.
6. Lock bytes, Git blobs, branch, clean HEAD, and origin are replayed internally.
7. Authorization schema, chronology, commit, lock, and protected false fields are replayed.
8. Output and external consumption roots are derived and matched exactly.
9. Consumption uses one `O_CREAT|O_EXCL` external receipt and fsyncs file and directory.
10. A copied/fabricated/mismatched context fails registered dispatch.
11. Receipt deletion or byte tampering fails registered dispatch.
12. An `AUTHORIZATION_CONSUMED` lifecycle identity is required for every registered dispatcher.

The implementation correctly describes this as official-result governance,
not adversarial Python or operating-system security.

## Atomic Transaction And Recovery Audit

All twelve checks pass:

1. Result artifacts, proof candidates, manifest, lifecycle, and completion receipt share one staging bundle.
2. The lifecycle is authoritative inside that bundle.
3. All result files are written before commit readiness.
4. Semantic replay completes before commit readiness.
5. Manifest and completion receipt are written before commit readiness.
6. The terminal event is `ATOMIC_PUBLISH_COMMIT_READY` and is written before rename.
7. Files and staging directories are fsynced before rename.
8. Publication performs exactly one `os.replace(staging, final)`.
9. There is no required callback, append, write, or replay after rename.
10. A valid final bundle after crash is recovered as success.
11. No final bundle exists if rename did not complete.
12. Terminal-without-final reconciliation cannot overwrite the primary exception.

The transaction tests inspect the commit method and inject pre-rename,
terminal-staging, and recovery failures. No test creates a registered C85T
result.

## Exception Precedence Audit

The primary exception survives lifecycle-append, cleanup, quarantine,
recovery, and reporting failures. Secondary errors are retained separately.
`FAILED` is appended once only for a nonterminal lifecycle. A terminal valid
final bundle is recovered as success; a terminal bundle without a final root
becomes a separate reconciliation blocker.

## Result-Semantic Replay Audit

All sixteen checks pass:

1. Exact scenario keys are exactly `S0` through `S10`.
2. Missing and extra scenario keys fail.
3. Scenario-specific required fields are enforced.
4. S10 replays `11/40`, `0`, `3/5`, and `13/40` exactly.
5. S8 rational LP certificate fields are required and replayed.
6. S6 has exactly 4,096 unique ordered replicate IDs.
7. S7 has exactly 4,096 unique ordered replicate IDs.
8. S6/S7 action ranges, binary indicators, and finite nonnegative regrets are checked.
9. S6/S7 aggregates are rederived from persisted arrays.
10. S9 has exactly 4,096 replicate IDs per design and 8,192 logical rows.
11. S9 actions are restricted to `{0,1,2,3}` and indicators are binary.
12. S9 digest CSV has exactly 4,096 unique rows with `<i8`, 51 L, and 46 H.
13. S9 SHA fields are lowercase 64-hex and combined identity is consistent.
14. The registered int64 stream is deterministically rerun under the consumed context.
15. Proof files and CSV rows cover exactly T1-T7 with matching candidate and statement hashes.
16. Result lock/auth/attempt/root/HEAD bindings and all protected zero counters replay.

Counts are derived from artifacts. The validator does not return hard-coded
success counts.

## Proof-Stage Separation

All six checks pass:

1. C85T can freeze proof candidates only.
2. Internal checks are labelled non-dispositive.
3. No same-process independent-review claim exists.
4. Automatic theorem transition is absent from the V3 coordinator.
5. T1-T7 formal statuses remain `OPEN`.
6. C85V requires separate PM approval and is not authorized by C85TR2.

## Scientific Boundary

All protected counters remain zero:

```text
registered S0-S10 draws:        0
registered scenario results:    0
registered MC replicate rows:   0
canonical proof candidates:     0
theorem-status transitions:     0
C85T authorization/consumption: 0 / 0
real project data access:        0
training / forward / GPU:        0 / 0 / 0
active acquisition:              0
C85V / C85E authorization:       0 / 0
manuscript work:                 0
```

No C85T V3 authorization record, result JSON, final bundle, or proof-candidate
directory exists in Git.

## Regression Audit

| Suite | Accepted result | Stderr |
|---|---:|---:|
| focused | 410 passed | 0 bytes |
| C65 | 1,021 passed, 1 skipped, 3 deselected | 0 bytes |
| C23 | 1,432 passed, 1 skipped, 3 deselected | 0 bytes |
| full OACI | 2,356 passed, 1 skipped, 3 deselected | 0 bytes |

The accepted skip is the finalized historical C78F test. The three standing
deselections are historical unauthorized C79 adapter tests. They are not
C85TR2 failures.

## Disclosed Non-Accepted Invocations

Two readiness-only errors are retained in the report:

1. The login Python 3.9 environment failed test collection before any test or
   registered path ran. Accepted runs used the locked Python 3.13 environment.
2. An initial lock-builder command supplied a mistyped implementation commit
   and failed before creating a lock or registry. The accepted builder used
   exact commit `d489bd428f1c39a6eb399c9697b983d0b143ec80`.

Neither consumed authorization or opened a registered seed.

## Git And Operations

At the lock audit:

```text
HEAD == origin/oaci:
  true

largest tracked file:
  21,936,073 bytes

Git file envelope:
  < 50 MiB PASS

raw data / model weights added:
  0 / 0
```

`squeue` showed one pre-existing user job, `897842`, named `bash`, running
`/bin/bash` on `cpu-high`. C85TR2 did not submit or alter it, and its visible
identity does not indicate C84/C85/OACI work. No `sacct` evidence is claimed.

## Final Boundary

The V3 lock is ready but not authorized:

```text
LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED
```

Only a new standalone `授权 C85T` issued after PM acceptance can bind lock
commit `b1a5ba3aca002de7e302fc375298cc69c1ed82a8`, lock SHA-256
`3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9`,
one committed V3 authorization record, one authorization ID, and one exact
content-addressed output root.

