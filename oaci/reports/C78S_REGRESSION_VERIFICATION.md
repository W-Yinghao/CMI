# C78S Regression Verification

All suites ran on Slurm `cpu-high` with 48 CPUs per job. All stderr logs are empty.

```text
focused:    43 passed
C65-C78S:  256 passed, 1 intentional conditional skip
C23-C78S:  663 passed, 1 intentional conditional skip
full OACI: 1591 passed, 1 intentional conditional skip
```

The sole skip in each broad suite is the intentional finalized-milestone guard:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

This reason was independently replayed with `pytest -rs` in Slurm job `893168`
(`256 passed, 1 skipped`). The focused C78S suite passed 43/43. No failed test
or warning was hidden.
