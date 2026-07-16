# C84SR3 Regression Verification

Final accepted commit: `4cbe49d68b280d90c3b49ec82cbbbf9e8df95ed9`.

| Suite | Job | Result | Pytest time | stderr |
|---|---:|---|---:|---:|
| focused | 898423 | 367 passed | 73.07 s | 0 bytes |
| C65 | 898425 | 853 passed, 1 skipped, 3 deselected | 97.07 s | 0 bytes |
| C23 | 898426 | 1,264 passed, 1 skipped, 3 deselected | 94.96 s | 0 bytes |
| full OACI | 898427 | 2,188 passed, 1 skipped, 3 deselected | 533.54 s | 0 bytes |

All accepted jobs used the locked `c84c-eeg2025-v3-exact` environment on
`cpu-high` with 48 CPUs, 96 GiB, and GPU 0. They were monitored with `squeue`;
`sacct` was not used. The sole skip is the finalized C78F execution test. The
three deselections are the established C79 tests whose premise is that the
future C79 authorization record is absent.

The first local attempt used the default Python and stopped during collection
because that environment has no `torch` distribution. The first locked-
environment focused attempt then exposed nine historical lifecycle assertions:
six lock-name sets stopped at V4, two V4 tests still assumed its authorization
was absent, and one V3 test assumed only one later implementation file had
changed. Those test-only assertions now describe the additive V5 lifecycle;
no V5 runtime byte, selector formula, threshold, frozen field, or scientific
contract changed after the V5 lock.

The complete full suite also passed locally before the independent Slurm replay:
2,188 passed, 1 skipped, and 3 deselected in 2,182.96 seconds.
