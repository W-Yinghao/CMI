# C81E Final Regression Verification

## Accepted Regression Head

All accepted jobs guarded the exact clean commit before invoking pytest:

```text
commit:      d88e9c93c9a373c5662d9dcdc01e0c28b220335d
worktree:    /home/infres/yinwang/CMI_AAAI_oaci
environment: /home/infres/yinwang/anaconda3/envs/eeg2025
partition:   cpu-high
allocation:  48 CPU, 96 GiB, GPU 0 per job
```

| Suite | Job | Result | stderr |
|---|---:|---|---:|
| focused C81E | 894970 | 48 passed | 0 bytes |
| C65-C81E | 894971 | 417 passed, 1 skipped, 3 deselected | 0 bytes |
| C23-C81E | 894972 | 828 passed, 1 skipped, 3 deselected | 0 bytes |
| full OACI | 894973 | 1,752 passed, 1 skipped, 3 deselected | 0 bytes |

The single conditional skip is exactly:

```text
oaci/tests/test_c78f_full_seed3_field.py:174:
C78F has already passed red-team and finalized
```

The three deselections are the historical C79P tests whose assertions describe
its old preauthorization state:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

No C81 test or registered path was skipped or deselected. The one-test increase
over C81R2 is
`test_C81E_blocker_consumes_authorization_and_forbids_same_protocol_rerun`,
which verifies that job `894958` consumed the authorization and that the
post-evaluation blocker cannot be rerun under the current protocol identity.

## Command Contract

Each Slurm wrapper required:

```text
HEAD == origin/oaci == d88e9c93c9a373c5662d9dcdc01e0c28b220335d
git status --porcelain-v1 == empty
PYTHONDONTWRITEBYTECODE=1
pytest -p no:cacheprovider -q -rs
```

The accepted logs, byte counts, hashes, durations, commands, and test counts are
recorded in `c81e_tables/final_regression_ledger.csv`. All four jobs exited
successfully. The scheduler-accounting database had one transient query failure
after completion; pytest logs and the Slurm control record for the final job
were available and agree on successful completion.

These regressions validate the fail-closed blocker state. They do not create a
C81-A/B/C/D scientific comparison and do not authorize a repair or C82.
