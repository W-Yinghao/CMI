# C85R Regression Verification

## Accepted Implementation

```text
repair protocol commit:
  03bb684e59e3432ae6f484c8c8a537213f52a6cd

implementation commit:
  360c422f3110d61cfef5d09fead78562bb52c497

environment:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python:
  3.13.7

GPU:
  0

scheduler evidence:
  squeue only

sacct used:
  false
```

No failed formal regression attempt preceded the accepted runs. Component
development tests passed before the clean implementation commit and are not
counted separately as accepted cumulative evidence.

## Leading-Numeric Suite Accounting

Both new suffix files are included in focused, C65, C23, and full accounting:

```text
test_c85r_synthetic_semantic_repair.py
test_c85r_protocol_lock.py
```

The two files add 30 tests. The leading-numeric parser still includes the
historical suffix milestones, including C34S.

## Accepted Runs

| Suite | Passed | Failed | Skipped | Deselected | Pytest runtime | stdout SHA-256 | stderr bytes |
|---|---:|---:|---:|---:|---:|---|---:|
| focused | 321 | 0 | 0 | 0 | 6.99 s | `5bf6e56ac6a143ef7e7467457effa6bf0ac2380b94bc8fadae452ca60d9060f9` | 0 |
| C65 cumulative | 932 | 0 | 1 | 3 | 74.39 s | `9f9a97c603362a7d1787541228d889ecf8f63e16e80af586f3deae0e77a31e39` | 0 |
| C23 cumulative | 1,343 | 0 | 1 | 3 | 104.33 s | `26a9ce1ec21b8a6134227338ee921b96907b53cc7110b1085732f93b2240bdfa` | 0 |
| full OACI | 2,267 | 0 | 1 | 3 | 315.50 s | `673dee5461dfde332ebee8507e4ea0996e746ac6a4a9fecf7c8bff794fe121f9` | 0 |

Every accepted stderr file has SHA-256:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The accepted skip is the historical finalized C78F red-team test. The three
accepted deselections are the standing C79P unauthorized-adapter checks in the
committed wrapper.

Relative to C85P, every suite adds exactly 30 tests:

```text
focused: 291 -> 321
C65:     902 -> 932
C23:   1,313 -> 1,343
full:  2,237 -> 2,267
```

## Runtime Boundary

The tests perform exact rational enumeration, schema replay, static import
audits, and covariance identities only. They do not generate the registered
4,096 synthetic replicates, complete a proof, change theorem status, read real
project arrays, run selectors/inference, train, forward, use GPU, or execute an
acquisition policy.

Post-run `squeue` showed zero active C84/C85/OACI jobs. `sacct` was not used.

## Disposition

```text
focused:        PASS
C65 cumulative: PASS
C23 cumulative: PASS
full OACI:      PASS
accepted stderr: EMPTY
```

