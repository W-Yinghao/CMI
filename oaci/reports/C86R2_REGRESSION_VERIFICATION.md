# C86R2 Regression Verification

## Execution Identity

```text
accepted evidence commit:
  b6c331d79ec9fae38dbc6c9379c19ef9ecac72a4

HEAD == origin/oaci during accepted runs:
  true

worktree clean during accepted runs:
  true

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

GPU:
  0
```

## Preserved Failed Attempts

The first readiness generator invocation stopped before artifact publication
because the implementation expected 89 deduplicated public candidate datasets,
while the exact frozen projection contained 79. The input consisted only of
safe public catalog fields. It opened no EEG, labels, predictions, active
result, or registered synthetic stream. The exact contract was corrected to
149 catalog candidate rows, 79 native datasets after mirror deduplication, and
82 interface rows after expanding the four native `ds007221` tasks. No cohort
eligibility decision changed.

Before the evidence commit, two pure contract tests and two 13-test focused
runs passed while the tables and public-source fields were being finalized.
They are development checks, not substitutes for the accepted clean-HEAD runs.

## Accepted Runs

| Suite | Result | Pytest time | stdout SHA-256 | stderr bytes |
|---|---|---:|---|---:|
| focused C86P/C86R/C86R2 | 51 passed | 0.50 s | `2e44a0ebb58eec8a06332c327b1d29685e68c871b29eb0a88a22c948e3bf81ac` | 0 |
| C65 cumulative | 1,103 passed, 1 skipped, 12 deselected | 102.41 s | `5ba3c86a323494d5fb7dd5b0308756da0c01774bb4298bcbc364eade31899b52` | 0 |
| C23 cumulative | 1,514 passed, 1 skipped, 12 deselected | 125.14 s | `c7385392b01e85b882f8f61bedea3b3720be6200a37e36d06eca0ac55dda0cf7` | 0 |
| full OACI | 2,489 passed, 1 skipped, 12 deselected | 333.91 s | `fa3939799a889ace4eae1e3c1dbeed722a866039af053a7efd4c9a77b9b3d3b3` | 0 |

Every accepted stderr file has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Deselection Accounting

The cumulative suites used the same twelve historical readiness-only absence
assertions registered by C86P and C86R:

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

Each assertion was true at its historical milestone and was superseded by a
later accepted scope-specific authorization. No C86P, C86R, or C86R2 test was
deselected. The sole skip remains the finalized historical C78F check.

## Semantic Coverage

```text
protocol chronology and SHA replay: PASS
adult age threshold exactly 18: PASS
Yang/Kumar fail-closed evidence: PASS
catalog snapshot and pagination identities: PASS
149 -> 79 native catalog deduplication: PASS
82 interface dispositions complete: PASS
all passing interfaces included: PASS
two adult cohorts / 53 target clusters: PASS
common field and resource arithmetic: PASS
stale V2 downstream guard: PASS
protected counters all zero: PASS
real-data lock and authorization absent: PASS
```

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
C86_ADULT_UNTOUCHED_MULTI_COHORT_ELIGIBILITY_RESOLVED_READY_FOR_C86LP_PROTOCOL_REVIEW
```
