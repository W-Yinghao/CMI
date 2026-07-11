# C80P Regression Verification

All four suites passed on `cpu-high` with 48 CPUs in the `eeg2025`
environment at commit `17946fa`. Slurm combined stderr with stdout; no error
diagnostics occurred in any passing job.

| Suite | Job | Passed | Failed | Skipped | Deselected |
|---|---:|---:|---:|---:|---:|
| focused | 893765 | 29 | 0 | 0 | 3 |
| C65-C80P | 893767 | 344 | 0 | 1 | 3 |
| C23-C80P | 893766 | 751 | 0 | 1 | 3 |
| full OACI | 893768 | 1,679 | 0 | 1 | 3 |

The conditional skip is the finalized C78F red-team guard. The three
deselections are C79P tests that freeze the pre-authorization state; their
post-authorization inverse remains covered by the accepted C79E tests. No C80
primary path was skipped: C80P has no real-data primary execution by design,
and all synthetic/registry paths are covered by the focused suite.
