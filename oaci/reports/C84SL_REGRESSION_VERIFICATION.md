# C84SL Regression Verification

## Accepted final suites

All accepted final suites ran against clean, pushed commit
`33075c97afd87f05d2856463c43be3246d83f95c` in the exact
`c84c-eeg2025-v3-exact` environment. Cumulative suites used `cpu-high`, 48 CPUs,
96 GiB, and GPU 0. Scheduler monitoring used `squeue`; no `sacct` claim is made.

| Suite | Execution | Result | Duration | stderr |
|---|---|---|---:|---:|
| focused C84SL | local exact environment | 55 passed | 36.04 s | 0 bytes |
| C65 cumulative | job `897816` | 813 passed, 1 skipped, 3 deselected | 63.44 s | 0 bytes |
| C23 cumulative | job `897817` | 1,224 passed, 1 skipped, 3 deselected | 159.64 s | 0 bytes |
| full OACI | job `897818` | 2,148 passed, 1 skipped, 3 deselected | 486.32 s | 0 bytes |

The cumulative suites used the corrected leading-numeric milestone parser, so
suffix milestones such as C34S and all C84SL tests were included.

The sole skip is:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

The three registered deselections are:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

They are historical C79 authorization-state checks and are not C84SL test
failures.

## Initial regression attempts

Initial attempts are preserved rather than overwritten:

| Attempt | Result | Disposition |
|---|---|---|
| initial focused | 49 passed, stderr 0 | implementation-level checks passed before the end-to-end repair |
| C65 job `897796` | 801 passed, 6 failed, 1 skipped, 3 deselected, stderr 0 | exposed six stale historical assertions that treated absence of any C84S lock as timeless |
| C23 job `897797` | 1,211 passed, 7 failed, 1 skipped, 3 deselected, stderr 0 | same six lifecycle failures plus one brittle exact NumPy cross-backend diagnostic assertion |
| full job `897798` | deliberately canceled near 73% after the same seven failures reproduced | stderr contains the 84-byte scheduler cancellation marker; not accepted |

Commit `8b49e99c417023d345837378f18d2a5952aa206c` corrected only lifecycle
semantics and the historical diagnostic test:

```text
historical-tree tests still require no C84S lock at their historical commit;
current-tree tests recognize the additive lock lifecycle;
the NumPy test binds the exact historical failed-artifact value and treats
synthetic cross-backend magnitude as finite diagnostic evidence.
```

## Intermediate replacement attempts

After the lifecycle correction:

| Attempt | Result | Disposition |
|---|---|---|
| replacement focused | 49 passed, stderr 0 | accepted for the then-current implementation only |
| C65 job `897809` | 807 passed, 1 skipped, 3 deselected, stderr 0 | superseded by the end-to-end repair |
| C23 job `897810` | 1,218 passed, 1 skipped, 3 deselected, stderr 0 | superseded by the end-to-end repair |
| full job `897811` | deliberately canceled during audit | a missing unified production Stage-C freeze path was discovered before completion; stderr contains the 84-byte cancellation marker |

No real target label, selector score, or scientific result had been accessed.
The audit therefore triggered an additive protocol and implementation repair,
not a post-outcome retry.

## End-to-end repair verification

The final implementation added the production analysis/result-freeze path,
executable S0--S20 scenarios, deterministic Q0 `FULL`, complete secondary table
writers, and V2 lock replay.

Final focused coverage includes:

```text
label-view isolation;
selection freeze;
Q0 nested acquisition and FULL determinism;
evaluation utility and aggregation;
Q1/Q2 max-T inference;
level/panel/seed/LOTO stability;
C84-A--E and C84-L1--L4 taxonomy;
18,608-row production Stage-C validation;
atomic complete-result publication;
V2 analysis-lock replay.
```

Synthetic S0--S20 calibration passed 21/21 through production public
entrypoints. Static isolation passed 16/16. Frozen external artifact replay
passed 3,888/3,888 SHA-256 checks.

## Final acceptance

No accepted suite has nonempty stderr. No required C84SL session remains
running. The final regression evidence supports:

```text
C84S_MULTIDATASET_LABEL_VIEWS_SELECTION_INFERENCE_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION
```

