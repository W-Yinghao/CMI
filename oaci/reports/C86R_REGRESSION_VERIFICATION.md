# C86R Regression Verification

## Execution Identity

```text
implementation commit:
  ffe1fa01b6e9395c5fde0f120d348897a270898e

effective-manifest binding commit:
  e4f344d65ec501398d74e43ded00bdcb967495f3

HEAD == origin/oaci during accepted runs:
  true

worktree clean during accepted runs:
  true

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python
```

## Preserved Failed Attempts

The initial focused C86R run produced `7 failed, 9 passed`. Every failure was
an assertion against a test-side field name that did not match the generated
versioned CSV schema. The generated arithmetic and policy decisions were not
changed to make the tests pass. Assertions were aligned to the actual schemas,
and two metadata fields were made explicit: the 1-of-2 adult-cohort failure in
the final cohort registry and the exact `1/65537` max-T resolution.

An initial timing wrapper also failed before pytest because `/usr/bin/time` is
not installed. It opened no protected input and produced no test result. The
accepted runs used the Bash `time` keyword with separate stdout and stderr
files.

## Accepted Runs

| Suite | Result | Pytest time | Wall time | stdout SHA-256 | stderr bytes |
|---|---|---:|---:|---|---:|
| focused C86P+C86R | 38 passed | 0.27 s | 0.621 s | `b0a0e7d69822c03d6ed422288000a55b9ce9ac094b1b4e92b98bd3511b440dd1` | 0 |
| C65 cumulative | 1,103 passed, 1 skipped, 12 deselected | 110.89 s | 112.518 s | `3e73949e74ebba75c2c47a3e6e7e28014dcdfa10a93bcdba0e25f048d8584e96` | 0 |
| C23 cumulative | 1,514 passed, 1 skipped, 12 deselected | 185.81 s | 187.627 s | `9724fd7e66c7470293d0c08f7eca96c35abcde0cb72da5e3ce06c4874075f15e` | 0 |
| full OACI | 2,476 passed, 1 skipped, 12 deselected | 2,246.82 s | 2,248.971 s | `cc54664fd7ab010a9467d060fed708c0704231d032348a70bec88652a63d356f` | 0 |

Every accepted stderr has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The full run was slower than the earlier C86P run but remained CPU-active and
completed without error.

## Deselection Accounting

The cumulative suites deselected the same twelve historical readiness-only
absence assertions already registered by C86P:

```text
3 C79 pre-authorization adapter assertions;
1 C85P no-C85-real-data-lock assertion;
1 C85R no-C85T-result assertion;
1 C85EP no-C85E-implementation assertion;
2 C85URP downstream-absence assertions;
1 C85UR1 no-C85U/C85E assertion;
1 C85TR2 no-C85T-result assertion;
1 C85VP no-C85V-result assertion;
1 C85EP2 no-C85E-execution assertion.
```

Those assertions were true at their historical milestones and were superseded
by later, separately authorized C85 stages. No C86P or C86R assertion was
deselected. The cumulative skip remains the finalized historical C78F check.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
C86_UNTOUCHED_COHORT_AGE_ACCESS_OR_INTERFACE_ELIGIBILITY_RECONCILIATION_REQUIRED
```
