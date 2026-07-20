# C84SR1 Regression Verification

## Accepted suites

All accepted suites ran against clean, pushed commit
`4774f72d4c2674fed409bab950bce8ce70df2264` in the exact
`c84c-eeg2025-v3-exact` environment. Jobs used the `cpu-high` partition and GPU
0. Scheduler monitoring used `squeue`; no `sacct` claim is made.

| Suite | Job | Result | Duration | stderr |
|---|---:|---|---:|---:|
| focused C84SR1 | `897836` | 19 passed | 3.25 s | 0 bytes |
| C65 cumulative | `897837` | 832 passed, 1 skipped, 3 deselected | 66.85 s | 0 bytes |
| C23 cumulative | `897838` | 1,243 passed, 1 skipped, 3 deselected | 93.31 s | 0 bytes |
| full OACI | `897839` | 2,167 passed, 1 skipped, 3 deselected | 490.33 s | 0 bytes |

The cumulative suites used the corrected leading-numeric milestone parser, so
suffix milestones such as C34S and every C84SR1 test were included.

The sole skip is:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

The three fixed deselections are historical C79 authorization-state tests:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

No C84SR1 implementation, lock, production-path, Q0, process-isolation or
result-materialization test was skipped or deselected.

## Superseded attempts

The first cumulative run found six historical tests whose current-tree
expectation listed only the V1 and V2 C84S locks. Historical-commit assertions
remained correct; only the current-tree set needed to recognize the additive V3
lock.

| Attempt | Result | Disposition |
|---|---|---|
| focused `897830` | 19 passed, stderr 0 | superseded because it preceded the clean pushed lock commit |
| C65 `897831` | 826 passed, 6 failed, 1 skipped, 3 deselected, stderr 0 | exposed six stale lock-enumeration assertions |
| C23 `897832` | 1,237 passed, 6 failed, 1 skipped, 3 deselected, stderr 0 | reproduced the same six assertions |
| full `897833` | cancellation requested after the deterministic cause was known | not accepted; the 84-byte stderr is the scheduler cancellation marker |
| focused `897835` | no pytest started | a mistyped expected commit was rejected by the submission guard; both logs are empty |

Commit `4774f72d4c2674fed409bab950bce8ce70df2264` changed only the six historical
test expectations and added the already-generated V3 lock/readiness artifacts.
It did not change a lock-bound implementation file or the V3 lock SHA-256.

## Acceptance

All accepted stderr files are empty. No C84SR1 regression job remains in
`squeue`. Regression evidence supports:

```text
C84S_REAL_EXECUTION_ORCHESTRATION_Q0_INTEGRATION_REPAIRED_AND_LOCKED_READY_FOR_FRESH_PI_AUTHORIZATION
```
