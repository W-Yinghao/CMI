# C79E Final Regression Verification

All four final suites passed on `cpu-high` with 48 CPUs in the `eeg2025`
environment at commit `3a02a0d`. This commit includes the final-report tests
and handoff. Slurm combined stderr with stdout; no error
diagnostics occurred in any passing job.

| Suite | Job | Passed | Failed | Skipped | Deselected |
|---|---:|---:|---:|---:|---:|
| focused | 893726 | 38 | 0 | 0 | 3 |
| C65-C79E | 893724 | 315 | 0 | 1 | 3 |
| C23-C79E | 893725 | 722 | 0 | 1 | 3 |
| full OACI | 893723 | 1,650 | 0 | 1 | 3 |

The one conditional skip is the finalized C78F red-team guard. The three
deselections are C79P tests that freeze the pre-authorization state; their
post-authorization inverse is covered by C79E authorized-runtime tests.

The first complete attempt is retained. Jobs `893710`, `893711`, and `893712`
each failed only two historical Mode-R artifact-compatibility assertions after
C79E used shared audit filenames. The additive repair restored the exact
Mode-R hashes and moved C79E risk/failure ledgers to `c79e_*` names. The two
focused failing assertions then passed 2/2 before the complete rerun. No
scientific table, estimand, gate, result, field artifact, or external payload
changed.

A third complete run was required because three final-report tests were added
after the compatibility rerun. The table above is that final-HEAD run; all
earlier attempts remain in `c79e_regression_attempt_ledger.csv`.
