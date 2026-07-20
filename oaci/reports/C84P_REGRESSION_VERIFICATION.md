# C84P Regression Verification

## Commit Under Test

All jobs guarded a clean canonical worktree and exact remote identity before
pytest started:

```text
commit:      edfeefb36c81dda8420e40a63d6f0b915ef6b4da
HEAD:        edfeefb36c81dda8420e40a63d6f0b915ef6b4da
origin/oaci: edfeefb36c81dda8420e40a63d6f0b915ef6b4da
environment: /home/infres/yinwang/anaconda3/envs/eeg2025
partition:   cpu-high
allocation:  48 CPU / 96 GiB / GPU 0 per job
```

| Suite | Job | Result | Pytest runtime | Slurm runtime | Stderr |
|---|---:|---|---:|---:|---:|
| focused C84P | 895316 | 28 passed | 0.50 s | 00:00:02 | 0 bytes |
| C65-C84P | 895317 | 514 passed, 1 skipped, 3 deselected | 59.85 s | 00:01:04 | 0 bytes |
| C23-C84P | 895318 | 921 passed, 1 skipped, 3 deselected | 101.64 s | 00:01:46 | 0 bytes |
| full OACI | 895319 | 1,849 passed, 1 skipped, 3 deselected | 263.49 s | 00:04:28 | 0 bytes |

All jobs completed with `ExitCode=0:0`. Every stderr file is empty and hashes
to the SHA-256 of an empty file. Exact commands, nodes, allocations, log paths,
byte counts and hashes are in `c84p_tables/regression_attempt_ledger.csv`.

## Skip Audit

`pytest -rs` reports one intentional conditional skip:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

The three explicit deselections are historical C79P assertions tied to its old
preauthorization state:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

No C84 test, metadata audit, synthetic scenario or protection check was skipped
or deselected.
