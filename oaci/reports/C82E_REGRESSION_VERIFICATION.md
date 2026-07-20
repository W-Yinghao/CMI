# C82E Regression Verification

## Accepted Commit

All accepted jobs guarded the clean canonical worktree and exact remote
identity before pytest started:

```text
commit:      d4c035d80de8c1ed1892f83d470296e89de74a06
HEAD:        d4c035d80de8c1ed1892f83d470296e89de74a06
origin/oaci: d4c035d80de8c1ed1892f83d470296e89de74a06
environment: /home/infres/yinwang/anaconda3/envs/eeg2025
partition:   cpu-high
allocation:  48 CPU, 96 GiB, GPU 0 per job
```

| Suite | Job | Result | Pytest runtime | stderr |
|---|---:|---|---:|---:|
| focused C82 | 895215 | 43 passed | 1.75 s | 0 bytes |
| C65-C82 | 895221 | 460 passed, 1 skipped, 3 deselected | 30.64 s | 0 bytes |
| C23-C82 | 895222 | 871 passed, 1 skipped, 3 deselected | 91.45 s | 0 bytes |
| full OACI | 895218 | 1,795 passed, 1 skipped, 3 deselected | 726.71 s | 0 bytes |

The counts match C82P exactly because C82E changed reports and immutable result
artifacts, not the locked implementation or test suite.

## Skip And Deselection Audit

`pytest -rs` reports one intentional conditional skip:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

The three explicit deselections remain the historical C79P tests whose
assertions describe its old preauthorization state:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

No C82 test or registered analysis path was skipped or deselected.

## Preserved Diagnostic Attempts

The first C65/C23 enumeration jobs (`895216`, `895217`) selected the correct
suite and passed, but five nonnumeric `test_c*.py` filenames caused shell
arithmetic warnings, leaving 540-byte stderr files. They are not accepted.

The first warning-free correction (`895219`, `895220`) required an underscore
after the numeric milestone and therefore omitted suffix milestones such as
C78F, C79E, and C80R. Its 308/715 counts are incomplete and not accepted.

The final leading-numeric parser included both numeric and suffix milestones,
recovered the exact 460/871 counts, and produced empty stderr. All eight
attempts, commands, outputs, hashes, and dispositions are retained in
`c82e_tables/regression_attempt_ledger.csv`.

## Log Identity

All four accepted stderr files are empty and hash to the SHA-256 of an empty
file. Complete stdout/stderr paths, byte counts, SHA-256 values, Slurm jobs,
allocations, and attempt dispositions are recorded in the machine ledger.
