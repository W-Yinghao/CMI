# C84C Regression Verification

All regression jobs ran CPU-only in `c84c-eeg2025-v3-exact` against result
commit `2f541e526deb79091ad164b0d37419941e6f662b`.

| Suite | Job | Result |
|---|---:|---|
| focused | 895528 | 112 passed |
| C65 | 895530 | 598 passed, 1 skipped, 3 deselected |
| C23 | 895531 | 1,009 passed, 1 skipped, 3 deselected |
| full | 895529 | 1,933 passed, 1 skipped, 3 deselected |

The conditional skip is
`test_c78f_full_seed3_field.py::test_c78f_full_field_schema_and_counts`, whose
reason is that C78F already passed red-team and finalized. The three explicit
deselections are historical C79 authorization-state tests whose fixture assumes
that no later execution authorization record exists. They do not conceal a C84
registry path or engineering check.

Every regression stderr file is empty. Logs, hashes, node observations and
allocations are recorded in `c84c_tables/regression_verification.csv`.
