# C83P Regression Verification

## Accepted Commit

All jobs guarded a clean canonical worktree and exact remote identity before
pytest started:

```text
commit:      c927b3a80bc1b75ed4d9ec0d7d2460342574ffe2
HEAD:        c927b3a80bc1b75ed4d9ec0d7d2460342574ffe2
origin/oaci: c927b3a80bc1b75ed4d9ec0d7d2460342574ffe2
environment: /home/infres/yinwang/anaconda3/envs/eeg2025
partition:   cpu-high
allocation:  48 CPU / 96 GiB / GPU 0 per job
```

| Suite | Job | Result | Pytest runtime | Stderr |
|---|---:|---|---:|---:|
| focused C83P | 895253 | 26 passed | 1.07 s | 0 bytes |
| C65-C83P | 895254 | 486 passed, 1 skipped, 3 deselected | 53.35 s | 0 bytes |
| C23-C83P | 895255 | 897 passed, 1 skipped, 3 deselected | 88.04 s | 0 bytes |
| full OACI | 895256 | 1,821 passed, 1 skipped, 3 deselected | 722.14 s | 0 bytes |

All four Slurm jobs completed with `ExitCode=0:0`. Every stderr file is empty
and hashes to the SHA-256 of an empty file.

## Skip And Deselection Audit

`pytest -rs` reports one intentional conditional skip:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

The three explicit deselections are historical C79P tests whose assertions
describe its old preauthorization state:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

No C83 test or evidence-freeze path was skipped or deselected. Exact commands,
job IDs, allocations, counts, logs, byte counts, and hashes are recorded in
`c83p_tables/regression_attempt_ledger.csv`.

