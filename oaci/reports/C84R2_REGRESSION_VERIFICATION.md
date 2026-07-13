# C84R2 Regression Verification

All four suites ran at verification commit `ecd1f5b0bbb83214f6cc4ce488742feb181cb325` (which contains C84C V2
lock commit `270fbb0d9f47f9bf6a2888ee58fd7ca6eadff0ea`) in the dedicated exact CPU
environment. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, the corrected
leading-numeric suite parser, and the established three C79P deselections were fixed.

| Suite | Job | Passed | Skipped | Deselected | Stderr |
|---|---:|---:|---:|---:|---|
| focused | 895362 | 90 | 0 | 0 | EMPTY |
| c65 | 895363 | 576 | 1 | 3 | EMPTY |
| c23 | 895364 | 987 | 1 | 3 | EMPTY |
| full | 895365 | 1911 | 1 | 3 | EMPTY |

The conditional skip, where present, is the finalized C78F field test. Every accepted
stderr file is empty. No GPU or real-data execution was requested.
