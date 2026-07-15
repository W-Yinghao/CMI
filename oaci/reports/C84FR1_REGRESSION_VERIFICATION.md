# C84FR1 Regression Verification

All suites ran at commit `f3ef6312427d6b60160ee89679e038d22055feff` in
`c84c-eeg2025-v3-exact` on `cpu-high` with 48 CPUs, 96 GiB, and no GPU.

| Suite | Job | Result | Pytest time | Observed node | stderr |
|---|---:|---|---:|---|---:|
| focused | 896544 | 242 passed | 4.30 s | nodecpu01 | 0 bytes |
| C65 | 896545 | 728 passed, 1 skipped, 3 deselected | 52.16 s | nodecpu02 | 0 bytes |
| C23 | 896546 | 1,139 passed, 1 skipped, 3 deselected | 112.62 s | nodecpu02 | 0 bytes |
| full OACI | 896547 | 2,063 passed, 1 skipped, 3 deselected | 272.22 s | nodecpu01 | 0 bytes |

The sole skip is
`test_c78f_full_seed3_field.py::test_no_final_report_exists_at_protocol_stage`:
C78F is finalized. The three deselections are the established historical C79
preauthorization-state checks:

- `test_real_execution_fails_closed_without_future_authorization_record`
- `test_show_binding_contract_is_the_only_unauthorized_adapter_command`
- `test_unauthorized_command_does_not_import_training_or_EEG_modules`

No C84FR1 test was skipped or deselected. The focused suite includes
`test_c84fr1_target_stage_repair.py` and passed all 15 C84FR1 tests. Application
summaries were observed in stdout and every job subsequently disappeared from
`squeue`; `sacct` was not used.

All stderr SHA-256 values are the empty-file digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
