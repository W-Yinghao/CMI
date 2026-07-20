# C84P Multi-Dataset Protocol Readiness

## Final Gate

```text
C84_DATASET_CHANNEL_EVENT_RESOURCE_OR_PROTOCOL_RECONCILIATION_REQUIRED
```

## Blocking Finding

The requested primary montage contains 21 channels. Official MOABB 1.5.0
metadata and loader source expose:

```text
Lee2019_MI:  20 / 21, FCz absent
Cho2017:     21 / 21
PhysionetMI: 21 / 21
```

Lee provides `Fz`, but the protocol explicitly forbids substitution,
interpolation and silent channel reduction. Therefore C84 cannot yet lock an
executable common primary interface. An additive, availability-only PM decision
is required before C84C. No real-data outcome was accessed, so the design can be
reconciled prospectively.

## Chronology

```text
C83P accepted base:                2ecc8efd49d6b9d18b50eae3811be8f2ac4cfa25
C84 metadata protocol/blocker:     7cf702e
C84 synthetic implementation:     bd42f71
C84 guarded regression entrypoint: edfeefb
```

The metadata protocol was committed and pushed before the synthetic
implementation. No C84 real-data adapter or execution lock was created.

## Protocol Identities

```text
external protocol SHA-256:
  ebfe9933ac838af22cc6553f81ba87d806996e99ce33158c7a5c30b9a1f5e824

canary protocol SHA-256:
  bacc511dec01c2141470e689e62b6664089ab2ce7b78255b46acd14446cbfffd

field protocol SHA-256:
  cbb515772ad4257b59e64a499c0af909c6bfabe54c7a2c3f3226e4cccd6d6f15

scientific-analysis protocol SHA-256:
  3c6810f99c1e69cd4e0758ff3ea2ca81799d06c0d18dc7d29e4d128a3ef4590c
```

All three stage protocols are non-operative and explicitly require a future
scope-specific execution lock after reconciliation.

## Locked Design

The metadata-only design freezes deterministic SHA-256 partitions with 16
panel-A subjects, 16 panel-B subjects and 22/20/76 targets for Lee/Cho/Physionet.
Each source panel is split deterministically into 12 training and 4 audit
subjects. All source-panel, source-target and target overlaps are zero.

The task mapping is left-hand versus right-hand imagery only. Physionet subject
88, executed runs and bilateral hand/feet runs are excluded prospectively; Lee
online unlabeled runs are excluded. The temporal interface is half-open `[0,3)`
at 160 Hz with exactly 480 samples. Trial IDs are split by a class-stratified
hash and remain metadata/dependence keys only.

Subject to montage reconciliation, the fixed-zoo arithmetic remains:

```text
candidate units:               1,944
training phases:                  72
target contexts:                 944
candidate-context evaluations: 76,464
```

The C82 selector identities and formulas are unchanged. Cross-dataset A/B
requires the same fixed method across Lee, Cho and Physionet. Target is the
principal within-dataset cluster; panel, seed and level are repeated factors.

## Synthetic And Resource Audit

```text
S0-S14 plus auxiliary checks: 20 / 20 PASS
focused C84P:                 28 passed
C65-C84P:                    514 passed, 1 skip, 3 deselected
C23-C84P:                    921 passed, 1 skip, 3 deselected
full OACI:                 1,849 passed, 1 skip, 3 deselected
all regression stderr:          0 bytes
```

The conservative planning estimate is 56.873925 GPU phase-hours, below the
250-hour envelope. Combined download plus derived storage is 630 GiB, below the
2 TiB envelope. These are planning estimates only; C84P downloaded no data and
allocated no GPU.

## Protected State

```text
real EEG arrays loaded:          0
real labels read:                0
dataset downloads:               0
training/forward/re-inference:   0
GPU jobs:                        0
real candidate units:            0
C84 execution locks:             0
C84C/F/S authorization consumed: 0
same-label oracle access:        0
BNCI2014_004 access:             0
manuscript drafting:             0
```

## Required Repair

PM must choose and prospectively register an availability-only common-montage
resolution. That repair must create a new operative protocol hash and new,
separate C84C/C84F/C84S locks. The current broad authorization statement does
not bypass the unresolved input contract or authorize an object that does not
yet exist.
