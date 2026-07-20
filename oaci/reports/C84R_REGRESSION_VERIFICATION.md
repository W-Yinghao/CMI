# C84R Regression Verification

All suites tested execution-lock commit `4eaad36cafefb2645f1d5c6e393ae5a51ff33af9` in the established CPU-only
`eeg2025` environment (48 CPU, 96 GiB, GPU 0). The corrected leading-numeric parser
includes suffix milestones and restores the four C34S nodes omitted by C84P.

| Suite | Job | Passed | Skipped | Deselected | Stderr |
|---|---:|---:|---:|---:|---|
| focused | 895347 | 56 | 0 | 0 | EMPTY |
| c65 | 895348 | 542 | 1 | 3 | EMPTY |
| c23 | 895349 | 953 | 1 | 3 | EMPTY |
| full | 895350 | 1877 | 1 | 3 | EMPTY |

The one conditional skip is `test_c78f_full_seed3_field.py:174`: C78F already
passed red-team and finalized. The three deselections are the established C79P
historical authorization-state tests. Exact commands, hashes and log paths are
recorded by this report's source ledger and all stderr files are empty.
