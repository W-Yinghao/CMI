# C85EP2 Regression Verification

## Context

```text
validated commit:
  48e177c9914003202cc75cefb4a98832ea8250c3

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

declared envelope:
  cpu-high / 32 CPU / 128 GiB / 0 GPU
```

Runs were local executions of the lock-bound Slurm-compatible wrapper. No
`sacct` evidence is claimed. Each accepted run required clean `oaci` HEAD equal
to `origin/oaci`.

## Accepted Runs

| Suite | Result | Pytest time | stdout SHA-256 | stderr bytes |
|---|---:|---:|---|---:|
| focused | 395 passed, 2 deselected | 9.59 s | `b52e14d94285989f6c08cb59d79b67ca4e7ab537ed388499ad0a8f60d5068ebe` | 0 |
| C65 | 1,104 passed, 1 skipped, 11 deselected | 88.12 s | `f182e6ef88741f5dac125eeb708c58337274d951131dd62396197ab0f0621cde` | 0 |
| C23 | 1,515 passed, 1 skipped, 11 deselected | 124.93 s | `68acd5c06075716ecdc29b193d9c3d9085abb3d9f2980ab85ae18b040fa31cf1` | 0 |
| full OACI | 2,439 passed, 1 skipped, 11 deselected | 329.68 s | `35f09decf8c8fdccc754d07a25f50f1e70507fd597405d0cfb140ed86b3e4141` | 0 |

Every accepted stderr has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Skip And Deselection Accounting

The single cumulative skip is finalized historical C78F. Three standing C79
unauthorized-adapter assertions and six historical absence assertions were
deselected by the wrapper. Two additional pre-C85V absence assertions were
supplied explicitly for accepted cumulative reruns after the first C65 attempt
identified them. No C85EP2 test was skipped or deselected.

## Preserved Initial Attempts

The pre-lock focused attempt produced `4 failed / 18 passed`; the failures were
resolved without weakening the scientific contract. The initial C65 run is
retained at:

```text
/home/infres/yinwang/CMI_AAAI/c85ep2_regression_logs/c65-48e177c9.out
```

It produced `1,104 passed / 2 failed / 1 skipped / 9 deselected`; its stdout
SHA-256 is
`c89c50191b38a5938246dde85535db2eff6e5cb8b3c759374678be43de8f9908`.
Both failed tests asserted that accepted C85V authorization/results did not yet
exist. The rerun deselected only those superseded assertions.

## Coverage

The accepted suites cover C85U identity and semantic replay, exact 944 x 81 and
18,432 / 8,749,056 arithmetic, direct-path isolation, raw-gap versus
standardized-risk separation, exact-collapse/T3 guards, stochastic Q0,
target-equal aggregation, fractional CVaR, applicability nulls, theorem
transfer guards, lock byte/Git replay, authorization fail-closed behavior, and
atomic publication failure injection.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
REAL_C85E_EXECUTIONS_ZERO
C85E_LOCK_READY_NOT_AUTHORIZED
```
