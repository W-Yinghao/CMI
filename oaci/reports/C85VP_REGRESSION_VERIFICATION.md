# C85VP Regression Verification

## Execution Context

```text
repository commit:
  3c732489407ebca7603e5fb65d03c1ae25d046b6

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

GPU:
  0

partition / requested CPU / memory:
  cpu-high / 48 / 96 GiB
```

Jobs were monitored with `squeue` and retained stdout/stderr files. No `sacct`
evidence is used or claimed.

## Accepted Runs

| Suite | Job | Result | Pytest time | stdout SHA-256 | stderr bytes | stderr SHA-256 |
|---|---:|---:|---:|---|---:|---|
| focused | 899937 | 395 passed | 13.19 s | `a8f502b253465c1567a74e4abda62eec1ceeb5accfa9a56feb97a5684f1e1ef6` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C65 | 899949 | 1,040 passed, 1 skipped, 4 deselected | 122.39 s | `533d3f82da6371caa30b2eca952b4511782e7e8ed353585556eba4894a6fc49f` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C23 | 899950 | 1,451 passed, 1 skipped, 4 deselected | 185.11 s | `bdfba97a293f98e65bddd79a98a53d2243eb3b2bea46f118e86e091736bcf749` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| full OACI | 899951 | 2,375 passed, 1 skipped, 4 deselected | 501.73 s | `fa42a0b2b037bbdcd2b7d224eb9c7c5f26074d3e999084098a3530f7a71e2c0f` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The focused suite includes all three new C85VP test files. C65, C23 and full
discover them through the leading-numeric milestone parser.

## Skip And Deselection Accounting

The one skip is the finalized historical C78F test:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

Three deselections are the standing C79 unauthorized-adapter tests. The fourth
is the accepted C85T post-execution exclusion:

```text
oaci/tests/test_c85tr2_lock.py::
  test_no_authorization_result_proof_or_status_transition_exists
```

That test asserts that the C85T V3 authorization record is absent. C85T V3 has
already been explicitly authorized and accepted, so the readiness-only
assertion is inapplicable. The same exclusion is documented in
`C85T_REGRESSION_VERIFICATION.md`. No C85VP test was deselected.

The accepted replacement runs passed this exact fourth deselection through
`PYTEST_ADDOPTS`; the committed C85VP wrapper and all lock-bound bytes remained
unchanged after lock creation.

## Preserved Non-Accepted Attempts

| Job | Suite | Disposition | stdout SHA-256 | stderr bytes | Reason |
|---|---|---|---|---:|---|
| 899938 | C65 | non-accepted | `89c3c1dbcf7cf8a58520691617db5dce6850108ebc6b8d4a1e7fc03365b3d1eb` | 0 | 1 stale C85TR2 readiness assertion failed after 1,040 passes |
| 899939 | C23 | non-accepted | `7c610f02f80f2975e4c6fe5d5521e10c2a7dad3fa280e33af0454d3dcd47a077` | 0 | same assertion failed after 1,451 passes |
| 899940 | full | cancelled non-accepted | `7d278cfc342643fab98815520295760b212f1150b42e254c7a8064554233e698` | 84 | cancelled after the deterministic historical failure was confirmed; Slurm cancellation marker retained |

The cancelled stderr SHA-256 is
`a7457d8696fb0021b57cec12365d0495eb43c37d31caf0b331681bb7bffd7412`.
These attempts did not alter the repository, lock, candidates, or formal
statuses.

## C85VP Test Coverage

The new focused tests cover:

```text
Stage-A candidate-path exclusion;
Stage-B release only after Stage-A freeze;
candidate and statement hash drift;
forbidden generator/Monte Carlo/real-data imports;
T2 exact rational replay;
T4 factor-of-two and TV boundaries;
T6 CVaR boundaries;
T7 sigma-zero, empty-set, tie and multiple-optimum cases;
T5 missing-decoder OPEN behavior;
finite-only versus general proof status;
single-rename atomic publication and failure injection;
lock, runtime registry and external identity replay;
zero authorization, registered review and status transition.
```

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
NO_REGISTERED_C85V_REVIEW_EXECUTED
NO_MONTE_CARLO_RERUN
T1_T7_REMAIN_OPEN
```

