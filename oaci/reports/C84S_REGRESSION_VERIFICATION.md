# C84S Regression Verification

## Accepted Runs

All accepted suites ran CPU-only in
`/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact`. Scheduler state
was monitored with `squeue`; `sacct` was not used.

| Suite | Job | Commit | Passed | Skipped | Deselected | Pytest time | stderr |
|---|---:|---|---:|---:|---:|---:|---:|
| focused C84 | 898593 | `9c3a864f` | 367 | 0 | 0 | 75.32 s | 0 bytes |
| C65 cumulative | 898594 | `9c3a864f` | 853 | 1 | 3 | 108.64 s | 0 bytes |
| C23 cumulative | 898619 | `c2f92f65` | 1,264 | 1 | 3 | 165.47 s | 0 bytes |
| full OACI | 898621 | `c2f92f65` | 2,188 | 1 | 3 | 488.13 s | 0 bytes |

All jobs requested 48 CPUs, 96 GiB, GPU 0 and the `cpu-high` partition. The
C65/C23 suites used the leading-numeric milestone parser, including suffixed
milestones such as C34S and all C84S/C84SR files.

The sole skip is:

```text
oaci/tests/test_c78f_full_seed3_field.py::test_c78f_full_field_schema_and_counts
```

Its registered reason is that C78F has already passed red-team and finalized.
The three cumulative deselections are:

```text
oaci/tests/test_c79p_post_seed3_protocol.py::test_real_execution_fails_closed_without_future_authorization_record
oaci/tests/test_c79p_post_seed3_protocol.py::test_show_binding_contract_is_the_only_unauthorized_adapter_command
oaci/tests/test_c79p_post_seed3_protocol.py::test_unauthorized_command_does_not_import_training_or_EEG_modules
```

Those tests freeze a historical no-later-authorization state and do not conceal
a C84 selector, label-view, inference, result-freeze or reporting failure.

Accepted log identities:

| Suite | stdout SHA-256 | stderr SHA-256 |
|---|---|---|
| focused | `25213c205335340255e10b4a95f6998a5b980198c314313811522c26e736d6f5` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C65 | `73920aed168e47cdc1cc5a4b47ec8fc0185265c4076ca5c82a4ed3b1e2e25716` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C23 | `f774f52f46cdd355f362276f5c2813ff12709e9765861a793c26386066ec648a` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| full | `5329126213fdc8a2746e9613e0f796128cd33029598fb1f1fbe5bc0d58b62e48` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Preserved Initial Attempts

Jobs `898595` and `898596` were submitted while report files were being
written. They reached the regression script after the worktree became dirty
and were rejected before pytest by the exact clean-worktree guard.

After the initial report commit, jobs `898615` and `898616` encountered the
same guard during the immediate compute-node visibility window. Their stdout
is empty and stderr contains only:

```text
C84L1C regression refused: worktree is not clean
```

Read-only diagnostic job `898617` then recorded on the compute node:

```text
HEAD   = c2f92f65c310aa895f6a2bcb060df07a6fbf475b
origin = c2f92f65c310aa895f6a2bcb060df07a6fbf475b
git status --porcelain = empty
stderr = 0 bytes
```

C23 and full were then resubmitted independently as accepted jobs `898619`
and `898621`. No scientific or runtime implementation changed between the
rejected preflights and accepted runs.

Regression logs are retained outside Git at:

```text
/home/infres/yinwang/CMI_AAAI/c84s_v5_regression_logs
```
