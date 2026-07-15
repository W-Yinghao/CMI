# C84F Regression Verification

All post-execution regression suites ran CPU-only in
`c84c-eeg2025-v3-exact` against authorization commit `fc40914d` plus the
uncommitted final-report artifacts. The suites were launched concurrently; all
processes exited zero and every stderr file is empty.

| Suite | Scope | Result | Pytest time |
|---|---|---|---:|
| focused | three C84FR2 test modules | 30 passed | 1.48 s |
| C65 | 54 milestone files | 758 passed, 1 skipped, 3 deselected | 608.98 s |
| C23 | 97 milestone files | 1,169 passed, 1 skipped, 3 deselected | 512.97 s |
| full | `oaci/tests` | 2,093 passed, 1 skipped, 3 deselected | 1,634.05 s |

The conditional skip is
`test_c78f_full_seed3_field.py::test_c78f_full_field_schema_and_counts`; its
reported reason is that C78F has already passed red-team and finalized.

The three explicit deselections are the established historical C79
authorization-state tests:

```text
test_c79p_post_seed3_protocol.py::test_real_execution_fails_closed_without_future_authorization_record
test_c79p_post_seed3_protocol.py::test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_c79p_post_seed3_protocol.py::test_unauthorized_command_does_not_import_training_or_EEG_modules
```

Those fixtures assume that no later execution authorization record exists and
are incompatible with the committed C84F authorization history. They do not
exclude a C84FR2 implementation, persistence, isolation or complete-field
contract test. The leading-numeric milestone parser remained in use for the C65
and C23 suites.

The immutable stdout, stderr and timing ledgers are retained outside Git under
`/home/infres/yinwang/CMI_AAAI/c84fr2_execution_logs/post_execution_regressions`.
