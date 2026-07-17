# C85EP Regression Verification

## Execution Context

```text
repository commit:
  f1574aec0738841ba3f52bbd7f6fc93204403e45

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

partition / CPU / memory / GPU:
  cpu-high / 48 / 96 GiB / 0
```

Each Slurm wrapper required `HEAD == origin/oaci` and a clean worktree before
starting pytest. Jobs were monitored with `squeue`; no `sacct` evidence is used
or claimed.

## Accepted Runs

| Suite | Job | Result | Pytest time | stdout SHA-256 | stderr bytes | stderr SHA-256 |
|---|---:|---:|---:|---|---:|---|
| focused | 900092 | 384 passed | 8.71 s | `188ddd3feca730818eaf9094c01e800d9436c597531fecc04e91993e97117e0a` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C65 | 900093 | 1,048 passed, 1 skipped, 5 deselected | 80.38 s | `55f3ec1e38593bd0d6e777acdec74dda95f61e7ca6eb04eb2a86d5778e19693b` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C23 | 900094 | 1,459 passed, 1 skipped, 5 deselected | 111.40 s | `8fef8343330e902e2d4ad21d68efa4199c3566fc84c963735d5f71b67ecdef14` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| full OACI | 900095 | 2,383 passed, 1 skipped, 5 deselected | 306.73 s | `0cffaeb0466252aae1c38e1159e7297bb2240fa3d56fcd10e2fd262f5b33e509` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The focused C85EP test file also passed independently before and after the
implementation commit:

```text
9 passed
```

## Skip And Deselection Accounting

The one skip is the finalized historical C78F test:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

Three deselections are the standing C79 unauthorized-adapter tests. Two more
are historical readiness-only absence assertions invalidated by accepted later
milestones:

```text
oaci/tests/test_c85tr2_lock.py::
  test_no_authorization_result_proof_or_status_transition_exists

oaci/tests/test_c85vp_execution_lock.py::
  test_c85vp_has_no_authorization_result_or_status_transition
```

C85T and C85V were subsequently authorized and accepted, so these historical
absence assertions are not applicable to current cumulative replay. No C85EP
test was deselected.

## C85EP Coverage

The new tests cover:

```text
protocol hash and commit chronology;
C85V theorem-scope addendum and frozen statuses;
manifest-only hard availability failure;
exact A1-A6 evidence rows;
metadata-only identity access classes;
epsilon/tau/CVaR grids and target-equal aggregation;
forbidden imports and protected paths;
absence of C85E lock, authorization and analysis implementation;
immutability of C84-D, C84-L4 and C85 theorem statuses.
```

## Preserved Attempts

There were no failed or cancelled C85EP regression attempts. All four submitted
jobs completed successfully and all accepted stderr files are empty.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
FROZEN_INPUT_BLOCKER_FAIL_CLOSED
NO_C85E_LOCK_OR_EXECUTION
```
