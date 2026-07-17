# C85V Regression Verification

## Execution Context

```text
repository commit:
  c5a818021a9c5d9ecc4cd661be84eb4e9efacbf1

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

partition / requested CPU / memory:
  cpu-high / 48 / 96 GiB

GPU:
  0
```

Jobs were monitored with `squeue` and retained stdout/stderr. No `sacct`
evidence is used or claimed.

## Accepted Runs

| Suite | Job | Result | Pytest time | stdout SHA-256 | stderr bytes | stderr SHA-256 |
|---|---:|---|---:|---|---:|---|
| focused | 900003 | 394 passed, 1 deselected | 9.12 s | `b6f398edb0c157c12256b48834ea94e512b8722963bd4ead0a9a77f9d01f73e4` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C65 | 900004 | 1,039 passed, 1 skipped, 5 deselected | 123.57 s | `7b54bbe4362c5260d09c79beca4c76210effd50028bd7027de43f233f7c726f0` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C23 | 900005 | 1,450 passed, 1 skipped, 5 deselected | 109.99 s | `1333e9931bc7b11eb971e2ec85f2d165f784ce87f4ebc54354ef2c70362e8817` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| full OACI | 900006 | 2,374 passed, 1 skipped, 5 deselected | 307.64 s | `43f14361ecb122e21fd51de245f20a46e8da7474dbbbb8bdd09470ee84798bed` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Skip And Deselection Accounting

The one skip is the finalized historical C78F test:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

Three cumulative deselections are the standing C79 unauthorized-adapter tests.
Two additional cumulative exclusions are lifecycle-specific:

```text
oaci/tests/test_c85tr2_lock.py::
  test_no_authorization_result_proof_or_status_transition_exists

oaci/tests/test_c85vp_execution_lock.py::
  test_c85vp_has_no_authorization_result_or_status_transition
```

The first asserts that the already accepted C85T authorization/result is
absent. The second asserts that C85V authorization/result is absent. Both are
readiness-only assertions superseded by explicit direct authorization and
frozen execution evidence. Focused includes the second test but not the first,
so it records one deselection.

The exclusions were passed through `PYTEST_ADDOPTS`; no committed wrapper,
test, protocol, implementation, or lock byte changed.

## Coverage Retained

The accepted suites still execute all applicable C85VP tests covering:

```text
Stage-A candidate-path exclusion;
Stage-B release only after Stage-A freeze;
candidate and statement hash drift;
forbidden generator/Monte Carlo/real-data imports;
T2/S10 exact rational replay;
T4 factor-of-two and TV boundaries;
T6 CVaR boundaries;
T7 sigma-zero, empty-set, tie and multiple-optimum cases;
T5 missing-decoder OPEN behavior;
finite-only versus general proof status;
single-rename atomic publication and failure injection;
lock, runtime-registry and external identity replay.
```

In addition, the production `validate_complete_bundle` function replayed the
authorized external C85V result against the exact lock/auth/attempt/root
identity after publication.

## Preserved Attempts

All four post-C85V runs passed on their first accepted invocation. No failed or
cancelled regression attempt occurred in this stage.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
AUTHORIZED_C85V_RESULT_SEMANTIC_REPLAY_PASS
NO_MONTE_CARLO_RERUN
NO_REAL_DATA_OR_ACTIVE_ACQUISITION
```
