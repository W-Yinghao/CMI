# C85UR1 Regression Verification

## Execution Context

```text
repository commit:
  672670d05e9d7adfbe12673d4a64bfd499413162

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

partition / CPU / memory / GPU:
  cpu-high / 48 / 128 GiB / 0
```

The Slurm wrapper required clean `oaci` HEAD equal to `origin/oaci` before
pytest. Jobs were monitored with `squeue`; no `sacct` evidence is used or
claimed.

## Accepted Runs

| Suite | Job | Result | Pytest time | stdout SHA-256 | stderr bytes |
|---|---:|---:|---:|---|---:|
| focused | 900419 | 396 passed | 10.62 s | `cf54d1ecc079293cb82222df101af5321f2baee4e676817bad34db3adfa10459` | 0 |
| C65 | 900420 | 1,088 passed, 1 skipped, 5 deselected | 128.09 s | `add07481dd159515198e6701b461c43f070fb4e214e0693a595945b7dc9b5990` | 0 |
| C23 | 900421 | 1,499 passed, 1 skipped, 5 deselected | 118.63 s | `c7ab075181ed7a5d850f9d9f77544333a67dfdb21181264fe8eada6603b847da` | 0 |
| full OACI | 900422 | 2,423 passed, 1 skipped, 5 deselected | 503.11 s | `7c5401d377e05cbde64f9909ca3a8ff39817dce7c381f1350727cfd08be5f9e8` | 0 |

Every accepted stderr file is empty and has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The four new C85UR1 test files independently passed after the V2 lock commit:

```text
21 passed
```

## Skip And Deselection Accounting

The single skip is the finalized historical C78F test:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
C78F has already passed red-team and finalized
```

Three deselections are standing C79 unauthorized-adapter tests. The other two
are historical readiness-only absence assertions superseded by accepted C85T
and C85V execution. No C85UR1 test was skipped or deselected.

## C85UR1 Coverage

The accepted runs cover:

```text
U1-only path definitions and dynamic open trapping;
semantic protected-replay receipt validation;
authorization/lock/attempt/root binding;
single-use U1 and U2 stage receipts;
direct-U2 and cross-attempt rejection before protected access;
V2 U1/U2 manifest and handoff replay;
18,432 endpoint and 8,749,056 finite-Q0-record arithmetic;
one-rename final acceptance transaction;
post-rename recovery and primary-exception precedence;
2 GiB U1 output enforcement;
V2 lock self-hash, 54 Git/byte bindings, chronology and protected counters;
immutability of C84-D, C84-L4 and all C85 theorem statuses.
```

## Preserved Attempts

The initial shadow run produced `12 passed / 4 failed`. It identified a
candidate-order fixture assumption, two fixture-parent omissions, and the V5
Lee/Cho/Physionet method-count arithmetic. The repaired shadow run passed
`16 / 16` before lock construction.

The first lock-builder call supplied an incorrect manually expanded
implementation SHA. It failed closed and wrote zero lock objects. The accepted
builder then used exact implementation commit
`5917cf54d33bbd5de906428bea5aaee22f45aabb` and wrote the three V2 lock
objects. All attempts preceded authorization and real protected access.

There were no failed or cancelled Slurm regression attempts. No active C85 job
remained after job 900422. The unrelated interactive job 897842 was not
created, modified, or counted as a C85 job.

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
REAL_C85U_PROTECTED_READS_ZERO
C85U_V2_LOCK_READY_NOT_AUTHORIZED
```
