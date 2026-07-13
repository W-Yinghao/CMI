# C82P Regression Verification

## Accepted Commit

All four accepted jobs guarded the clean canonical worktree and exact remote
identity before pytest started:

```text
commit:      6c6739c61d362bc33df6d8b016e4cda724772a62
HEAD:        6c6739c61d362bc33df6d8b016e4cda724772a62
origin/oaci: 6c6739c61d362bc33df6d8b016e4cda724772a62
environment: /home/infres/yinwang/anaconda3/envs/eeg2025
partition:   cpu-high
allocation:  48 CPU, 96 GiB, GPU 0 per job
```

| Suite | Job | Result | Runtime | stderr |
|---|---:|---|---:|---:|
| focused C82P | 895177 | 43 passed | 1.80 s | 0 bytes |
| C65-C82P | 895178 | 460 passed, 1 skipped, 3 deselected | 50.61 s | 0 bytes |
| C23-C82P | 895179 | 871 passed, 1 skipped, 3 deselected | 87.09 s | 0 bytes |
| full OACI | 895180 | 1,795 passed, 1 skipped, 3 deselected | 720.47 s | 0 bytes |

Each cumulative count is exactly the accepted C81E count plus the 43 C82 tests:

```text
C65:  417 + 43 =   460
C23:  828 + 43 =   871
full: 1752 + 43 = 1795
```

## Skip Audit

`pytest -rs` reports one intentional conditional skip:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

The three explicit deselections are the historical C79P tests whose assertions
describe its old preauthorization state:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

No C82 test or registered path was skipped or deselected. The focused suite
unconditionally tests the missing-lock authorization guard even after the real
lock exists by injecting temporary absent paths.

## Log Identity

The complete commands, log paths, byte counts, SHA-256 identities, allocations,
and results are recorded in `c82p_tables/regression_attempt_ledger.csv`. All four
stderr files are empty and hash to the SHA-256 of an empty file. Job `895180`
also replayed in `scontrol` as `COMPLETED`, `ExitCode=0:0`; complete pytest
summaries and empty stderr provide the accepted result records for all jobs.
