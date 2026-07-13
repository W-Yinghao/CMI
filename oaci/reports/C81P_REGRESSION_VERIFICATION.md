# C81P Regression Verification

## Accepted Regression Head

All accepted evidence was generated from the clean canonical worktree at:

```text
commit:      e347a06edf9fdcabd999cb848301d7d6b025c36c
worktree:    /home/infres/yinwang/CMI_AAAI_oaci
environment: /home/infres/yinwang/anaconda3/envs/eeg2025
partition:   cpu-high
allocation:  48 CPU, 96 GiB, GPU 0 per job
```

Each accepted job guarded the exact commit, `origin/oaci` identity, and clean
worktree before collecting tests.

| Suite | Job | Result | stderr |
|---|---:|---|---:|
| focused C81P | 894763 | 43 passed | 0 bytes |
| C65-C81P | 894764 | 412 passed, 1 skipped, 3 deselected | 0 bytes |
| C23-C81P | 894765 | 823 passed, 1 skipped, 3 deselected | 0 bytes |
| full OACI | 894766 | 1,747 passed, 1 skipped, 3 deselected | 0 bytes |

The single conditional skip is the finalized C78F guard at
`test_c78f_full_seed3_field.py:174`. The three deselections are the historical
C79P tests whose assertions intentionally describe its preauthorization state:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

No C81 registry path was skipped or deselected. C81 real-data paths were not
executed because C81P is readiness-only and no C81E authorization record exists.

## Attempt Ledger

All attempts are retained in
`oaci/reports/c81p_tables/regression_attempt_ledger.csv` with command status,
commit, environment, counts, log paths, byte counts, and hashes.

The superseded attempts fall into four transparent groups:

```text
894743 / 894750: passed an earlier test set, superseded after replay coverage grew
894744-894745: invalid newline file-list quoting in the Slurm wrapper
894746 / 894751: cancelled after the test set changed
894754-894757: fail-closed guard rejected an incorrect manually expanded SHA
894758-894761: fail-closed guard rejected accidental CSV line-ending changes
```

The line-ending changes were restored byte-for-byte before the accepted jobs;
the analysis-lock hash replay again matched
`b383707f58063c10f719194a995ab34094f6dcefe08c1e71837644db83dc94f1`.
No invalid or partial run is counted as regression evidence.
