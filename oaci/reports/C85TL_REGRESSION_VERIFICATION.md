# C85TL Regression Verification

## Accepted Lock State

```text
protocol commit:
  7e8ffdffcbd8aef5a59e6bfa9a2fe0c5aa20a28f

implementation commit:
  dad9d39cccf02771d4e643c0649fd66ab660a1c0

execution-lock commit:
  9d414ebb889b2cfc3fefa19fa98d7ea5ca9fd691

execution-lock SHA-256:
  4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991

environment:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python / NumPy runtime:
  3.13.7 / 2.4.4

GPU:
  0

scheduler evidence:
  squeue only

sacct used:
  false
```

No failed formal regression attempt preceded the accepted runs. An initial
four-suite pass at lock commit `68101202` was retained externally, then
superseded because review found that the lock's non-scientific
`created_at_utc` value was later than its Git commit. The timestamp alone was
corrected under commit `9d414ebb`; all four suites were rerun on that final
operative lock. Component and shadow development tests are not counted as
additional formal suites.

## Suite Accounting

The leading-numeric parser includes both C85TL files in focused, C65, C23, and
full accounting:

```text
oaci/tests/test_c85t_shadow_execution.py
oaci/tests/test_c85tl_execution_lock.py
```

Node contribution:

```text
shadow execution:
  16

chronology / environment / execution lock:
  11

total:
  27
```

Relative to C85R, all suites increase by exactly 27:

| Suite | C85R | C85TL | Delta |
|---|---:|---:|---:|
| focused | 321 | 348 | 27 |
| C65 cumulative | 932 | 959 | 27 |
| C23 cumulative | 1,343 | 1,370 | 27 |
| full OACI | 2,267 | 2,294 | 27 |

Historical suffix milestones, including C34S, remain included.

## Accepted Runs

| Suite | Passed | Failed | Skipped | Deselected | Runtime | stdout SHA-256 | stderr bytes |
|---|---:|---:|---:|---:|---:|---|---:|
| focused | 348 | 0 | 0 | 0 | 6.92 s | `a64b0717dd8aef5495c0c28643a0fedb1a4c2604818d316a783dd8751c626984` | 0 |
| C65 cumulative | 959 | 0 | 1 | 3 | 68.12 s | `99bd4822d7d7ea59f972eb3929ce95816a9b358ebee7475bb9a16c6ab77d9579` | 0 |
| C23 cumulative | 1,370 | 0 | 1 | 3 | 95.62 s | `523fedbd550bffd7b28574fc51c966c6437cc0b896201a9e3f2fa6b3fd3e66b5` | 0 |
| full OACI | 2,294 | 0 | 1 | 3 | 296.85 s | `fa6a10d4744d848faacedb6eaeeaa75f464c75d8e8dbc35c5fca1fb5c11f387b` | 0 |

Every accepted stderr file has SHA-256:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

## Skip And Deselection Explanation

The accepted skip is:

```text
oaci/tests/test_c78f_full_seed3_field.py:174
reason:
  C78F has already passed red-team and finalized
```

The three deselections are explicitly fixed in the committed wrapper:

```text
test_real_execution_fails_closed_without_future_authorization_record
test_show_binding_contract_is_the_only_unauthorized_adapter_command
test_unauthorized_command_does_not_import_training_or_EEG_modules
```

They are standing C79P unauthorized-adapter nodes, not C85TL failures or
unreported skips.

## C85TL Coverage

The accepted tests cover:

```text
protocol-before-implementation chronology;
C85P/C85R/V2 hash replay;
Python/NumPy runtime and dual metadata replay;
eleven bound NumPy file identities;
PCG64DXSM low64 seed replay;
shadow normal and Rademacher raw-byte replay;
registered RNG denial without authorization;
shadow exact CVaR and rational LP;
shadow S6/S7-style 4,096-replicate MC;
shadow S9-style paired 4,096-replicate MC;
analytic-versus-MC variance separation;
proof transition denial without independent PASS;
T5 OPEN preservation;
atomic success publication;
three injected atomic failure points;
106 bound repository object hashes and blobs;
runtime-bound registry identity;
lock ancestry and discovered lock commit;
missing authorization fail-closed;
absence of result/proof/C85E/active artifacts;
static empirical-stack import isolation.
```

The tests do not call a registered S0-S10 stream, render a canonical proof,
transition a theorem status, access real project data, run active acquisition,
or authorize C85E.

## Post-Run State

```text
active C84/C85/OACI jobs through squeue:
  0

accepted stderr files nonempty:
  0

registered scenario results created:
  0

proof files created:
  0

authorization records created:
  0
```

## Disposition

```text
focused:        PASS
C65 cumulative: PASS
C23 cumulative: PASS
full OACI:      PASS
accepted stderr: EMPTY
```
