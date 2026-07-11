# C78S Regression Verification

All suites ran on Slurm `cpu-high` with 48 CPUs per job. All stderr logs are empty.

```text
focused:    43 passed
C65-C78S:  256 passed, 1 intentional conditional skip
C23-C78S:  663 passed, 1 intentional conditional skip
full OACI: 1591 passed, 1 intentional conditional skip
```

The conditional skip is the route-absent compatibility branch in the C78S test
module; the focused suite, executed after the route lock existed, passed 43/43.
No failed test or warning was hidden.
