# C85TR1 Regression Verification

## Accepted V2 Lock State

```text
repair protocol commit:
  46442b281d61d00a575fae17685648b749659263

repair protocol SHA-256:
  9c0a7084a7ddd83ef96b8d7f95faf89138829729c0acc5c3d6baeb0ef87ab13d

implementation commit:
  f17e25d0d8dc117f7973f90743e07139eeb0c1e1

C85T V2 execution-lock commit:
  920c5540a6ae157b77f2acb36f227bfdc172110b

C85T V2 execution-lock SHA-256:
  0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719

historical lock SHA-256:
  4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991

environment:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python / NumPy runtime:
  3.13.7 / 2.4.4

CPU / GPU:
  48 / 0 for formal regression

scheduler evidence:
  squeue only

sacct used:
  false
```

All four accepted formal runs used a clean worktree with
`HEAD == origin/oaci == 920c5540a6ae157b77f2acb36f227bfdc172110b`.
No failed formal regression attempt preceded them. Component tests and shadow
calibration runs are disclosed separately and are not counted as formal suite
runs.

## Committed Wrapper Controls

The committed wrapper is:

```text
oaci/slurm_c85tr1_regression.sh
```

It requires:

```text
branch state:
  HEAD == origin/oaci

worktree:
  clean

test enumeration:
  committed leading-numeric suite parser

PYTHONHASHSEED:
  0

OMP / MKL / OpenBLAS / NumExpr threads:
  1

pytest cache and basetemp:
  external /tmp paths

empty suite:
  fail closed
```

The wrapper requests `cpu-high`, 48 CPUs, 96 GiB RAM, no GPU, and a one-day
wall envelope. The regression itself does not open a registered S0-S10 stream.

## New Test Accounting

C85TR1 adds three test files:

```text
oaci/tests/test_c85tr1_execution_guard.py
oaci/tests/test_c85tr1_replicate_persistence.py
oaci/tests/test_c85tr1_lock.py
```

Node contribution:

| Area | Nodes |
|---|---:|
| authorization, capability, lifecycle, proof separation | 8 |
| RNG bytes, intervals, replicate persistence, atomicity | 11 |
| protocol/lock chronology, replay, isolation | 8 |
| **total** | **27** |

The leading-numeric parser includes all three files in focused, C65, C23, and
full accounting. Relative to the accepted C85TL suites, every count increases
by exactly 27:

| Suite | C85TL | C85TR1 | Delta |
|---|---:|---:|---:|
| focused | 348 | 375 | 27 |
| C65 cumulative | 959 | 986 | 27 |
| C23 cumulative | 1,370 | 1,397 | 27 |
| full OACI | 2,294 | 2,321 | 27 |

## Accepted Formal Runs

| Suite | Passed | Failed | Skipped | Deselected | Runtime | stdout SHA-256 | stderr bytes |
|---|---:|---:|---:|---:|---:|---|---:|
| focused | 375 | 0 | 0 | 0 | 10.44 s | `8c6d8ba8cc0576b02044988eccd677373784d1cdded3905a370bd93839792a03` | 0 |
| C65 cumulative | 986 | 0 | 1 | 3 | 79.89 s | `3205b3a6ad287d6d654007f4069f196bef2195dcdd2dd57be3cae8909a0889a7` | 0 |
| C23 cumulative | 1,397 | 0 | 1 | 3 | 106.10 s | `62933e9b2077118aa08296eb9b598b81e00a1c5973ca7e28f3f6dae7617039fe` | 0 |
| full OACI | 2,321 | 0 | 1 | 3 | 315.15 s | `b18c863cff635d18a5a85ccedb333a9677bc2a04668856296cb4f58c75b94e8b` | 0 |

Every accepted stderr file has SHA-256:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

No accepted stdout contains a traceback, runtime error, failed-test marker, or
error summary.

## Skip And Deselection Accounting

The one accepted skip is historical:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
reason:
  C78F has already passed red-team and finalized
```

The three explicit deselections are standing C79P unauthorized-adapter nodes:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

They are fixed in the committed wrapper. They are not hidden failures, dynamic
skips, C85TR1 exceptions, or post-result exclusions.

## Core Guard Coverage

The accepted tests prove:

```text
protocol commit precedes implementation and V2 lock;
historical lock SHA remains exact and non-operative;
V2 lock sidecar replays;
133 bound repository bytes and Git blobs replay;
runtime registry exactly matches the lock;
required 16-table set is complete and nonempty;
no V2 authorization/result/proof/status artifact exists;
operative modules have no static execution token;
operative modules import no torch, MNE, or MOABB;
int64 and historical uint8 Rademacher bytes differ;
operative S9 draw explicitly requests numpy.int64;
raw and clipped probability intervals replay;
S6/S7 saved arrays reproduce every aggregate;
S9 saved arrays and 4,096 raw-digest rows replay;
missing/duplicate/reordered replicate IDs fail;
object and nonfinite arrays fail persistence;
same authorization cannot be consumed twice;
authorization cannot migrate to another output root;
fabricated string/object/None cannot unlock registered RNG;
capability cannot migrate to another attempt/root;
lifecycle is append-only and ordered;
failure location and primary exception replay;
automatic theorem transition is impossible;
all T1-T7 formal statuses remain OPEN;
complete shadow manifest validates exact row counts;
three injected atomic failures leave no final root.
```

## Shadow Versus Registered Execution

Shadow tests use only:

```text
SHADOW_NORMAL_A
SHADOW_RADEMACHER_A
SHADOW_RADEMACHER_B
```

They exercise 4,096-replicate persistence schemas but do not use S0-S10
identifiers and cannot create a registered result. Current protected counts are:

```text
registered S0-S10 draws:        0
registered scenario results:    0
canonical proof candidates:     0
theorem-status transitions:      0
authorization records:          0
authorization consumptions:     0
```

## Post-Lock Byte Audit

After the formal suites, the lock-bound implementation was compared against
the committed V2 lock. No implementation difference remains. A direct
lock-specific verification on the exact bytes passed:

```text
27 passed in 1.00 s
```

The V2 lock file SHA and sidecar both replayed as
`0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719`.

## Scheduler And Repository State

```text
active C84/C85/OACI jobs through squeue:
  0

sacct queried:
  false

GPU jobs submitted:
  0

registered execution jobs submitted:
  0

HEAD/origin divergence during accepted runs:
  0

accepted nonempty stderr files:
  0
```

## Protected Boundary

The regressions did not read EEG, label roots, target logits, source artifacts,
model states, checkpoints, or empirical result arrays. They did not run a
selector, empirical inference, training, forward pass, GPU path, active
acquisition, C85V, C85E, or manuscript path.

## Disposition

```text
focused:        PASS
C65 cumulative: PASS
C23 cumulative: PASS
full OACI:      PASS
accepted stderr: EMPTY
lock-byte replay: PASS
```

Regression evidence supports the readiness gate only. It does not authorize
C85T and is not a synthetic scientific result.
