# C84L1C Regression Verification

All four regression jobs ran CPU-only in `c84c-eeg2025-v3-exact` against
commit `ffdd7504d6fb0a31f8ad619f590cd7bbe4a3b4b8`.

| Suite | Job | Result | Elapsed | Node | Stderr |
|---|---:|---|---:|---|---:|
| focused | 896121 | 183 passed | 4.41 s | not observed before short job left `squeue` | 0 bytes |
| C65 | 896122 | 669 passed, 1 skipped, 3 deselected | 49.45 s | nodecpu02 | 0 bytes |
| C23 | 896123 | 1,080 passed, 1 skipped, 3 deselected | 111.01 s | nodecpu02 | 0 bytes |
| full | 896124 | 2,004 passed, 1 skipped, 3 deselected | 266.53 s | nodecpu01 | 0 bytes |

Each job requested `cpu-high`, 48 CPUs, 96 GiB and no GPU. Completion was
tracked with `squeue` and the complete pytest stdout; `sacct` was not used.
Scheduler exit codes are therefore not asserted. Every job left `squeue`,
printed a complete passing pytest summary, and produced empty stderr.

The conditional skip is
`test_c78f_full_seed3_field.py::test_c78f_full_field_schema_and_counts`, whose
reason is that C78F already passed red-team and finalized. The three explicit
deselections are historical C79 authorization-state tests whose fixtures assume
that no later authorization record exists. They do not conceal a C84L1C
engineering path.

The corrected leading-numeric suite parser remains active. The C23 suite
includes the C34S suffix milestone tests and the new C84L1C result-freeze tests.
Exact log identities are recorded in
`c84l1c_tables/regression_verification.csv`.
