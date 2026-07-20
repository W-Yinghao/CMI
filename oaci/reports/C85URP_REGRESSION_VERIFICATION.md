# C85URP Regression Verification

## Execution Context

```text
repository commit:
  3b4fa48ee2d4f75ff8ba2191dc7d8593237dc82f

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

partition / CPU / memory / GPU:
  cpu-high / 48 / 128 GiB / 0
```

The Slurm wrapper required clean `oaci` HEAD equal to `origin/oaci` before
pytest. Jobs were monitored with `squeue`; no `sacct` evidence is used or
claimed.

## Accepted Runs

| Suite | Job | Result | Pytest time | stdout SHA-256 | stderr bytes | stderr SHA-256 |
|---|---:|---:|---:|---|---:|---|
| focused | 900238 | 394 passed | 10.38 s | `15c81521103cd69497e496365dc7a20140fb0bf33fdf83d7d9caa5e97fc059be` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C65 | 900239 | 1,067 passed, 1 skipped, 5 deselected | 127.68 s | `30f3b448c093f16f8d2f0f91a005ab7b9706c6e84e421cb0fc8a82a05e7c7a6a` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C23 | 900240 | 1,478 passed, 1 skipped, 5 deselected | 113.90 s | `cd45ca297f595e7a1e05c591fdbcc277863d79a72d9dc70663e94680c7e0b804` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| full OACI | 900241 | 2,402 passed, 1 skipped, 5 deselected | 314.65 s | `c9adba6526e42f216e85629d37dc3c643a0c13c0debbd351656d594cc551d3f5` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The four new C85URP files independently passed after the lock commit:

```text
19 passed
```

## Skip And Deselection Accounting

The one skip is the finalized historical C78F test:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

Three deselections are the standing C79 unauthorized-adapter tests. The other
two are historical readiness-only absence assertions superseded by accepted
C85T and C85V execution:

```text
oaci/tests/test_c85tr2_lock.py::
  test_no_authorization_result_proof_or_status_transition_exists

oaci/tests/test_c85vp_execution_lock.py::
  test_c85vp_has_no_authorization_result_or_status_transition
```

No C85URP test was deselected.

## C85URP Coverage

The new tests cover:

```text
81-candidate historical metric and utility replay;
midrank ties and first-index canonical argmax;
zero-spread standardized regret;
candidate and trial identity failure;
exact NPZ dtype/shape/digest persistence;
atomic complete-field publication and injected partial failure;
deterministic and frozen-chain Q0 endpoint replay;
missing-chain and historical endpoint mismatch failure;
U1/U2 static and subprocess isolation;
single-use O_EXCL receipt behavior;
metadata-only 944-context / 1,944-artifact registry;
lock self-hash, Git chronology, schemas, counts and protected counters;
immutability of C84-D, C84-L4 and C85 theorem statuses.
```

## Preserved Attempts

The initial readiness-table generation exposed one empty-list error-message
bug before protected access and before lock construction. It was corrected
prospectively; protected access counters remained zero. The first lock-builder
invocation supplied an incorrect implementation SHA and failed before writing
any lock object. The successful lock was then built from the exact clean HEAD.

There were no failed or cancelled Slurm regression attempts. All four accepted
stderr files are empty.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
REAL_C85U_EXECUTIONS_ZERO
C85U_LOCK_READY_NOT_AUTHORIZED
```
