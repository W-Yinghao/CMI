# C85T V3 Regression Verification

## Execution Context

```text
repository commit:
  b26b21f6b8378188dd59890c5701944c41fad823

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

Python version:
  3.13.7

NumPy runtime:
  2.4.4

GPU:
  0

regression Slurm job:
  899525

partition / requested CPU / memory:
  cpu-high / 48 / 96 GiB
```

The regression stage ran only tests. It did not rerun S0-S10, regenerate Monte
Carlo arrays, or modify proof candidates.

## Accepted Runs

| Suite | Result | Pytest time | Wall seconds | stdout SHA-256 | stderr bytes | stderr SHA-256 |
|---|---:|---:|---:|---|---:|---|
| focused | 409 passed, 1 deselected | 13.35 s | 14 | `b54f97aa2740b990a28900d75bbec3b41c84a7e39f2ed42c4dbd27e7f74ba337` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C65 | 1,020 passed, 1 skipped, 4 deselected | 79.05 s | 82 | `d3f5e8dccbac8d1152f2b39a941b7c56af62d2117f43d969581e5fadd488952d` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C23 | 1,431 passed, 1 skipped, 4 deselected | 111.41 s | 113 | `2a9df371d5b4370bd2d267bd2898deb3517b453591a086accf62be61f1cd1994` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| full OACI | 2,355 passed, 1 skipped, 4 deselected | 306.20 s | 310 | `6d8fcdcc7a3d7010016404a4c971c6e64b2b7d5c663ba98309a082ad370b5c2f` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Logs are retained at:

```text
/home/infres/yinwang/CMI_AAAI/c85t_v3_regression_logs/
```

## Skip And Deselection Accounting

The one skip is the finalized historical C78F test:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

Three deselections are the standing historical C79 unauthorized-adapter tests.
The fourth is:

```text
oaci/tests/test_c85tr2_lock.py::
  test_no_authorization_result_proof_or_status_transition_exists
```

That test is a readiness-only assertion that the V3 authorization record is
absent. After explicit authorization and successful execution, the assertion
is intentionally inapplicable. All remaining C85TR2 lock, transaction,
semantic-replay, authorization-context, and compatibility tests ran.

No lock-bound test file was modified after authorization.

## Non-Accepted Invocations

Two operational invocations failed before an accepted replay/test began:

1. A post-execution manifest replay command used login Python 3.9 and failed at
   import because historical `dataclass(slots=True)` requires the locked Python
   3.13 environment. It did not rerun any scenario or write any result. The
   accepted replay used the exact locked Python 3.13 interpreter.
2. The first regression `srun` command inherited stale
   `SLURM_JOB_ID=896072`. Slurm rejected it before allocation and before pytest
   started. The accepted run removed the stale variable and received job
   `899525`.

Neither event changed or regenerated the external result bundle.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
REGISTERED_RESULT_NOT_RERUN
PROOF_CANDIDATES_UNCHANGED
```

