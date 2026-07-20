# C85TR2 Regression Verification

## Environment

```text
repository commit:
  b1a5ba3aca002de7e302fc375298cc69c1ed82a8

python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

Python:
  3.13.7

NumPy runtime:
  2.4.4

GPU:
  0

PYTHONHASHSEED:
  0
```

All formal suites used `oaci/slurm_c85tr2_regression.sh`. The focused wrapper
additively includes the four C85TR2 test files without modifying the
V2-lock-bound leading-numeric parser. C65 and C23 discover them through the
existing leading-numeric parser.

## Accepted Runs

| Suite | Result | Pytest time | Wall seconds | stdout SHA-256 | stderr bytes | stderr SHA-256 |
|---|---:|---:|---:|---|---:|---|
| focused | 410 passed | 12.62 s | 13 | `4b2c18cfbcc638569d7e549e2c6e09695bc0a7d37443272955f806096cded43a` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C65 | 1,021 passed, 1 skipped, 3 deselected | 81.39 s | 84 | `73a5f61f4d06d9582a915ad2baa0716ef4659120113663fa9e1b6614cb06b749` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C23 | 1,432 passed, 1 skipped, 3 deselected | 113.61 s | 116 | `7644f5782299bd161a2f96c0dfade3c9c6581b40f51af95005fb4dc3d5d9343e` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| full | 2,356 passed, 1 skipped, 3 deselected | 322.22 s | 325 | `7de683c36b1ba13c961dcce55fb99d6f7dc64c962e86cc0e1d43fb40f0bb4fb4` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

External logs are retained under:

```text
/home/infres/yinwang/CMI_AAAI/c85tr2_regression_logs/
```

## Skip And Deselection Accounting

The accepted skip is:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

The standing deselections are the three historical C79 unauthorized-adapter
tests already excluded by prior cumulative wrappers. They are not C85TR2 tests.

## Focused C85TR2 Coverage

The four C85TR2 files contribute 35 direct tests covering:

```text
committed lock/authorization path replay;
O_EXCL single use;
context copy/fabrication/root mismatch;
receipt deletion and tamper;
internal RNG helper fail-closed behavior;
terminal-before-rename transaction;
post-rename recovery;
primary/secondary exception precedence;
exact S0-S10 key coverage;
S6/S7 replicate and aggregate semantics;
S9 action/dtype/count/digest semantics;
proof file/CSV/statement identity;
result lock/auth/attempt/root identity;
V3 lock and all 160 bound object identities;
zero authorization, draw, proof, and status transition.
```

## Non-Accepted Readiness Invocations

The initial login-Python collection attempt failed under Python 3.9 before test
execution. The accepted runs above use only the exact locked Python 3.13
environment and have empty stderr. A separate initial lock-builder invocation
with a mistyped full commit failed before writing lock artifacts; it is not a
regression run.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
NO_REGISTERED_C85T_EXECUTION
```
