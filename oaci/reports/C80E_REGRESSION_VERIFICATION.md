# C80E Regression Verification

## Accepted regression head

All accepted regression evidence was generated on the clean canonical worktree
at commit:

```text
HEAD == origin/oaci == ebc6afe33ac7f15648012d1916281df277190228
worktree: /home/infres/yinwang/CMI_AAAI_oaci
environment: /home/infres/yinwang/anaconda3/envs/eeg2025
Python: 3.13.7
pytest: 9.0.2
partition: cpu-high
allocation per job: 48 CPU, 96 GiB, GPU 0
```

Each accepted job failed closed unless `HEAD`, `origin/oaci`, and the clean
worktree all matched the recorded commit before test collection.

| Suite | Job | Result | Runtime | stderr |
|---|---:|---|---:|---:|
| focused C80 | 894660 | 54 passed | 00:00:04 | 0 bytes |
| C65-C80E | 894661 | 369 passed, 1 skipped, 3 deselected | 00:00:52 | 0 bytes |
| C23-C80E | 894662 | 776 passed, 1 skipped, 3 deselected | 00:01:38 | 0 bytes |
| full OACI | 894663 | 1,704 passed, 1 skipped, 3 deselected | 00:04:20 | 0 bytes |

The single conditional skip is
`test_c78f_full_seed3_field.py:174`: C78F had already passed red-team and
finalized. The three deselections are the registered historical C79P tests
whose assertions intentionally describe the preauthorization review state;
they do not conceal a C80 path:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

The focused suite included the C80 frontier, C80E historical blocker, and C80R
additive-repair tests. All five historical-blocker tests pass after replacing
the obsolete assertion that result artifacts can never exist with a chronology
test: the preserved preflight had zero outcome access, the repaired direct
authorization followed it, and the result freeze followed authorization.

## Superseded attempts

No failed or invalid regression attempt was hidden.

Jobs `894651`-`894654` invoked a historical script whose hard-coded `cd`
selected the stale clean worktree at `1c9fd01`. Job `894651` completed there;
the other three were cancelled as soon as the path mismatch was detected. None
is accepted as C80E evidence.

Jobs `894655`-`894658` correctly targeted report commit `212d864`. Focused job
`894655` and C65 job `894656` exposed the obsolete lifecycle assertion; C23 and
full jobs `894657`/`894658` were cancelled after that common cause was known.
Commit `ebc6afe` changed only that lifecycle regression and retained the
historical zero-outcome blocker evidence. All four accepted suites were then
rerun from scratch.

The complete attempt ledger and stdout/stderr hashes are in
`c80e_tables/regression_attempt_ledger.csv`. External logs remain outside Git
under `/home/infres/yinwang/CMI_AAAI/c80e_regression_logs` and
`/home/infres/yinwang/CMI_AAAI/c80r_regression_logs`.

## Verdict

```text
C80E_FINAL_REGRESSION_VERIFICATION_PASSED
```

The regression repair changed no scientific registry entry, budget, selector,
RNG stream, outcome, threshold, dependence rule, taxonomy, or frozen result
artifact.
