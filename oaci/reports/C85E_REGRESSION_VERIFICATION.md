# C85E Regression Verification

## Execution Context

```text
repository commit:
  c65402db892fb8a58a0592ca5e81a0ee871a88ce

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

partition / CPU / RAM / GPU:
  cpu-high / 32 / 128 GiB / 0
```

The four accepted suites ran against the exact clean authorization/execution
HEAD with `HEAD == origin/oaci`.

## Accepted Runs

| Suite | Job | Result | Pytest time | stdout SHA-256 | stderr bytes |
|---|---:|---|---:|---|---:|
| focused | 900803 | 394 passed, 3 deselected | 9.56 s | `184be63af4f8e26a729fd46e2b3b92d1ff3581f2f7cd155e55e309e9ff5732d9` | 0 |
| C65 | 900804 | 1,103 passed, 1 skipped, 12 deselected | 92.77 s | `c9da5134e31ea6464acc25c031cca47592a21813f86bcd455f35eb9552a412e5` | 0 |
| C23 | 900805 | 1,514 passed, 1 skipped, 12 deselected | 189.18 s | `d18cb6e356694256c7afaff85ec4100dfad848e4f5cf52bb710877869b948f1e` | 0 |
| full OACI | 900806 | 2,438 passed, 1 skipped, 12 deselected | 522.82 s | `4da968238a5d615187341b8c10d229cc5a4c5e41330e7cc5bf1848d99e0d0440` | 0 |

Every accepted stderr file has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The single skip is the finalized historical C78F test at
`oaci/tests/test_c78f_full_seed3_field.py:174`.

## Deselection Accounting

The cumulative suites explicitly deselected twelve historical readiness-only
assertions:

```text
3 C79 unauthorized-adapter assertions;
1 C85P assertion that no C85 real-data/active lock exists;
1 C85R assertion that no C85T result exists;
1 C85EP assertion that no C85E implementation/lock/authorization exists;
1 C85URP immutable-downstream-absence assertion;
1 C85URP no-C85U/C85E authorization assertion;
1 C85UR1 no-C85U/C85E authorization assertion;
1 C85TR2 no-C85T authorization/result assertion;
1 C85VP no-C85V authorization/result assertion;
1 C85EP2 no-C85E authorization/execution assertion.
```

Each assertion was correct at its historical readiness milestone and was
superseded only by a later accepted direct authorization. No current C85E
analysis, result-semantic, geometry, risk, or runtime-boundary test was
deselected. No test or lock-bound implementation byte was changed.

## Production Replay

The lock-bound validator independently replayed the frozen result after job
completion:

```text
29 / 29 manifest artifacts;
26 / 26 registered result tables;
21,607 CSV rows and POST_C84S_EXPLORATORY tags;
32 exact bundle files and no extras;
result/manifest/completion/lifecycle hashes;
authorization, lock, attempt and output-root linkage;
all protected counters zero;
C84-D/C84-L4 and all theorem statuses unchanged.
```

The validator returned `PASS`. No residual staging root or active C85E/
regression job remained at collection.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
C85E_PRODUCTION_SEMANTIC_REPLAY_PASS
C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_COMPLETE_C86_PROTOCOL_REVIEW_REQUIRED
C86_STILL_UNAUTHORIZED
```
