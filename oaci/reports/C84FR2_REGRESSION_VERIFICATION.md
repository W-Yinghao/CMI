# C84FR2 Regression Verification

All accepted suites ran at lock commit
`34eb5efad16f1f8a320b08435363939089c8037a` in the exact
`c84c-eeg2025-v3-exact` environment on CPU only.

| Suite | Result | Pytest time | stderr |
|---|---|---:|---:|
| focused C84FR2 | 30 passed | 1.26 s | 0 bytes |
| C65 cumulative | 758 passed, 1 skipped, 3 deselected | 39.16 s | 0 bytes |
| C23 cumulative | 1,169 passed, 1 skipped, 3 deselected | 74.32 s | 0 bytes |
| full OACI | 2,093 passed, 1 skipped, 3 deselected | 281.43 s | 0 bytes |

The first C65 invocation omitted the established three C79 preauthorization
deselections. It completed with 758 passed, one skip, and those three expected
failures in 44.85 seconds. That log is preserved. The replacement C65 command
used these exact node IDs:

- `test_real_execution_fails_closed_without_future_authorization_record`
- `test_show_binding_contract_is_the_only_unauthorized_adapter_command`
- `test_unauthorized_command_does_not_import_training_or_EEG_modules`

All are in `oaci/tests/test_c79p_post_seed3_protocol.py`. The sole skip in each
cumulative suite is
`test_c78f_full_seed3_field.py::test_no_final_report_exists_at_protocol_stage`:
C78F is already finalized. No C84FR2 node was skipped or deselected.

Accepted stdout SHA-256 values:

```text
focused  70cab4d98a33933dcf1a0b1ecead80a04de8c98b3ad242578677409d27a1cdd1
C65     22b684193d05cfd6a54bc65d3627d499ea1edc784ac7925e9a88af0fb73f0ce9
C23     fd2fe4b5dca62d8e0e65d85b8a1c433f782eb2664fc9a83ea20ac4231ba0f8b4
full    29a6778dc013091efc1e0b3c7de8f0161e9b3eaf666aeb272be7ef6e9b1bbe75
```

Every accepted stderr file has the empty-file SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Logs are external under
`/home/infres/yinwang/CMI_AAAI/c84fr2_regression_logs` and are not tracked in
Git. These regressions did not reload target X, run forward, allocate a GPU, or
access labels/scientific outputs.
