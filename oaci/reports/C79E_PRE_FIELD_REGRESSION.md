# C79E Pre-Field Regression

All jobs ran on `cpu-high` in the `eeg2025` environment at commit `1f2ab3b`.
No GPU or seed-4 data path was used.

| Suite | Job | Passed | Failed | Skipped | Deselected |
|---|---:|---:|---:|---:|---:|
| focused | 893536 | 23 | 0 | 0 | 3 |
| C65-C79E | 893539 | 300 | 0 | 1 | 3 |
| C23-C79E | 893540 | 707 | 0 | 1 | 3 |
| full OACI | 893541 | 1,635 | 0 | 1 | 3 |

The intentional skip is the already finalized C78F red-team test. The three
intentional deselections assert that the C79E authorization record is absent;
those state-specific tests remain frozen at C79P commit `f176a64`. Their
post-authorization inverse is covered by `test_c79e_authorized_runtime.py`.

These are pre-field regressions and will be rerun after the C79E scientific
result is frozen.

