# C80R Regression Verification

All final suites passed on the exact clean commit
`93d2099f14b8739089e640c0e6078f02ed5cc435` in the Slurm `cpu-high`
partition. Each job used 48 CPUs, 96 GiB RAM, no GPU, and
`/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python`.

| Suite | Job | Passed | Failed | Skipped | Deselected | Stderr |
|---|---:|---:|---:|---:|---:|---:|
| focused C80/C80E/C80R | 894616 | 53 | 0 | 0 | 3 | 0 bytes |
| C65-C80R | 894617 | 368 | 0 | 1 | 3 | 0 bytes |
| C23-C80R | 894618 | 775 | 0 | 1 | 3 | 0 bytes |
| full OACI | 894619 | 1,703 | 0 | 1 | 3 | 0 bytes |

The submitted command was `sbatch oaci/slurm_c80r_regression.sh <suite>`.
The script ran `python -m pytest -q -rs` with suite-specific test globs,
isolated `/tmp` pytest/cache directories, and single-threaded numerical
libraries.

The only skip was:

```text
oaci/tests/test_c78f_full_seed3_field.py:174:
C78F has already passed red-team and finalized
```

The three explicit deselections are historical C79P tests that require the
future C79E authorization record to be absent:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

That preauthorization state no longer holds because C79E was completed under
its own accepted authorization. Its postauthorization behavior remains covered
by the C79E suites. No C80R path was skipped or deselected.

Two pre-final validation repairs are retained in
`c80r_tables/regression_attempt_ledger.csv`. First, jobs 894593-894596 wrote to
node-local logs that were not durable; they were replaced without changing
tests. Second, the initial C65/C23 glob omitted C80E/C80R tests; jobs
894610/894611 corrected the glob before finalization. The full-suite glob had
already included those tests. Neither repair accessed a scientific outcome or
changed the scientific registry.

All final stdout hashes and empty-stderr hashes are recorded in the attempt
ledger. No regression job allocated a GPU or invoked the real-data adapter.
