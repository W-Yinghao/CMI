# C85U Regression Verification

## Execution Context

```text
repository commit:
  f4b05c3dbed962348efe9cab56374854120a3667

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

partition / requested CPU / memory / GPU:
  cpu-high / 48 / 128 GiB / 0
```

The accepted post-execution regressions ran against the exact clean
authorization/execution HEAD. Jobs were monitored with `squeue` and retained
stdout/stderr. A later `sacct` query was attempted but the accounting database
connection was unavailable; no `sacct` terminal evidence is claimed.

## Accepted Runs

| Suite | Job | Result | Pytest time | stdout SHA-256 | stderr bytes |
|---|---:|---:|---:|---|---:|
| focused | 900453 | 395 passed, 1 deselected | 10.90 s | `2b441fbc7a950e530c6fed01a1f13288e66d2148e81ae5b3b60bce362c2e3134` | 0 |
| C65 | 900454 | 1,087 passed, 1 skipped, 6 deselected | 127.49 s | `ae75c3860d1768e0647be7adfdce68b925f4b58da55bdd5dd5f574d1e9477b1d` | 0 |
| C23 | 900455 | 1,498 passed, 1 skipped, 6 deselected | 118.87 s | `76d9fbd4032a6b15845546ea826fc4b1be40d9e1c4e4ff666d967d018983c668` | 0 |
| full OACI | 900456 | 2,422 passed, 1 skipped, 6 deselected | 518.44 s | `e0f6a91354c33e603cb7fa6fdff73e4c29affec94b061108747c9ff11617423c` | 0 |

Every accepted stderr file has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Skip And Deselection Accounting

The single skip is the finalized historical C78F test:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

Three cumulative deselections are the standing C79 unauthorized-adapter tests.
Two are historical absence assertions superseded by accepted C85T and C85V
execution. The sixth is:

```text
oaci/tests/test_c85ur1_lock.py::
  test_no_authorization_real_execution_or_downstream_lock
```

That readiness-only test requires the now valid C85U V2 authorization record to
be absent. It was explicitly deselected after direct PI authorization. No
implementation, lock, or test byte was changed.

## Production Replay

In addition to pytest, production validators replayed:

```text
944 U1 context artifacts;
76,464 candidate utility rows;
U1 manifest, handoff, sidecars and 44,003,342-byte tree;
18,432 U2 historical method-context rows;
8,749,056 finite Q0 action records;
all U2 endpoint maxima exactly zero;
authorization receipt, lifecycle, completion and acceptance manifest;
all final protected counters equal zero.
```

## Preserved Attempts

The first dry-preflight shell command failed at Python parse time because of an
embedded f-string escaping error. It consumed no authorization and performed no
protected read. The corrected dry preflight passed before the one authorized
execution. Job 900451 succeeded on its only authorized attempt; no retry was
made.

All four regression jobs passed on their first accepted invocation. At final
collection there was no active C85U or C85 regression job.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
C85U_PRODUCTION_REPLAY_PASS
C85U_COMPLETE_FIELD_ACCEPTED_C85E_STILL_UNAUTHORIZED
```
