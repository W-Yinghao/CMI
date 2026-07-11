# C79P Regression Verification

All suites ran on the established Slurm `cpu-high` partition with 48 CPUs and
the `eeg2025` environment against commit `ccdbf49`. No GPU or seed-4 job was
submitted.

```text
focused     job 893311:   21 passed, 0 failed, 0 skipped,   0.95 s
C65-C79P    job 893312:  298 passed, 0 failed, 1 skipped,  53.39 s
C23-C79P    job 893310:  705 passed, 0 failed, 1 skipped,  66.47 s
full OACI   job 893313: 1633 passed, 0 failed, 1 skipped, 419.60 s
```

The one skip is intentional and identical in each cumulative suite:
`C78F has already passed red-team and finalized`. The focused C79P suite has no
skip. The Slurm scripts use the default combined stdout/stderr stream; all four
combined logs contain normal pytest progress and summaries with no error
diagnostics.

Commands, exact paths, commit, counts, skip reason, and allocation details are
recorded in `c79p_tables/regression_verification.csv`.
