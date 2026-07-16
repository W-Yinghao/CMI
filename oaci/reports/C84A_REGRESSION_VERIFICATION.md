# C84A Regression Verification

## Accepted Implementation

```text
implementation/report commit:
  4076459996e2feecc9b7fa3aa6c036932f59f30e

environment:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python:
  3.13.7

GPU:
  0

scheduler evidence:
  squeue plus persisted stdout/stderr

sacct used:
  false
```

## Preserved Initial Failure

The first interactive focused run used the shell-default Python 3.9.13. It
produced `255 passed, 1 failed` in 9.70 seconds. The only failure was the
historical C84R2 exact-environment check:

```text
observed Python:
  3.9.13

lock-expected Python:
  3.13.7

classification:
  REJECTED_WRONG_ENVIRONMENT_NOT_CODE_REGRESSION
```

The exact environment rerun produced `256 passed` in 15.46 seconds. This
failed attempt is retained here rather than omitted or counted as accepted.

## Accepted Slurm Runs

| Suite | Job | Passed | Failed | Skipped | Deselected | Runtime | stdout SHA-256 | stderr bytes |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| focused | 898806 | 256 | 0 | 0 | 0 | 7.94 s | `af2c677ef0322c110dc67971a0821ad947f97c706a2a98485e6262ed57920bc6` | 0 |
| C65 cumulative | 898807 | 867 | 0 | 1 | 3 | 112.91 s | `3485318effa1a12ba100f07ad066c8bb65b14d2c24a43d8fba49b64f7182a404` | 0 |
| C23 cumulative | 898808 | 1,278 | 0 | 1 | 3 | 170.70 s | `937fbe51d711049e6fb3d717b7f2484226d4f520a182f74c81dd2db9a453021f` | 0 |
| full OACI | 898809 | 2,202 | 0 | 1 | 3 | 479.80 s | `88dd07be7e8e5f7dd27428dd46349642bba2f74c778657fddc422d623e0e16a7` | 0 |

All four stderr files are empty and have SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The one accepted skip is the historical finalized C78F red-team test. The
three accepted deselections are the standing C79P unauthorized-adapter checks
declared in the regression wrapper.

Relative to the accepted C84S cumulative counts, C65, C23 and full each add
exactly the 14 C84A tests:

```text
C65:
  853 -> 867

C23:
  1,264 -> 1,278

full:
  2,188 -> 2,202
```

Post-run `squeue` showed zero active C84/C85 jobs. No regression read EEG,
label roots, target logits, source arrays, selector/Q0 execution paths, or
scientific inference inputs.

## Disposition

```text
focused:
  PASS

C65 cumulative:
  PASS

C23 cumulative:
  PASS

full OACI:
  PASS

accepted stderr:
  EMPTY
```
