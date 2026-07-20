# OACI EEG-DG Project Memory Through C84P

## Current State

```text
C83P accepted base:
  2ecc8efd49d6b9d18b50eae3811be8f2ac4cfa25

C84 metadata protocol/blocker commit:
  7cf702e

C84 synthetic implementation:
  bd42f71

C84 guarded regression entrypoint:
  edfeefb

C84P blocker/red-team report:
  a1c781c

C84P final gate:
  C84_DATASET_CHANNEL_EVENT_RESOURCE_OR_PROTOCOL_RECONCILIATION_REQUIRED
```

C83 remains an immutable BNCI2014_001 evidence freeze. C84P starts a separate
multi-dataset left/right motor-imagery external-validity branch and computes no
new scientific outcome.

## Metadata Decision

The requested primary channel allowlist has 21 channels. Metadata and locked
loader-source inspection found:

```text
Lee2019_MI:  FCz absent, 20 / 21 requested channels available
Cho2017:     21 / 21 available
PhysionetMI: 21 / 21 available
```

The protocol forbids interpolation, `Fz` substitution and silent montage
reduction. This is the sole open C84P blocker. It was found before dataset
download, raw-array opening, label access, training, forward or selector
computation. An additive availability-only protocol revision can therefore
resolve the design prospectively, but the present hashes cannot authorize a
canary.

## Frozen Metadata Design

Eligible populations are Lee 54, Cho 52 and Physionet 108 after prospective
exclusion of Physionet subject 88. The SHA-256 subject partition gives two
disjoint 16-subject source panels per dataset and target populations of 22, 20
and 76. Each source panel is split deterministically into 12 training and 4
source-audit subjects. All panel and source-target overlaps are zero.

The task includes left-hand versus right-hand imagery only. Lee online unlabeled
runs, Physionet execution/bilateral runs, non-EEG channels and all other classes
are excluded. The temporal contract is half-open `[0,3)` after the official cue,
resampled to 160 Hz and exactly 480 samples. Stable trial IDs are metadata and
dependence keys, never features.

Subject to channel reconciliation, the fixed zoo remains:

```text
panels:                         A / B
training seeds:                5 / 6
levels:                        0 / 1
candidates per zoo:            81
candidate units:               1,944
training phases:               72
target contexts:               944
candidate-context evaluations: 76,464
```

The C82 selector registry is unchanged: S1 is strict source; U7, U5, U11, U13,
U14 and U15 are the six primary zero-label methods; Q0 B=1 is the primary
labeled comparator. Common budgets are `[1,2,4,8,FULL]`; Lee/Cho 16 and 32 are
secondary only. Cross-dataset A/B requires the same fixed method in all three
datasets.

## Protocol Identities

```text
C84 external protocol:
  ebfe9933ac838af22cc6553f81ba87d806996e99ce33158c7a5c30b9a1f5e824

C84C canary protocol:
  bacc511dec01c2141470e689e62b6664089ab2ce7b78255b46acd14446cbfffd

C84F field protocol:
  cbb515772ad4257b59e64a499c0af909c6bfabe54c7a2c3f3226e4cccd6d6f15

C84S science protocol:
  3c6810f99c1e69cd4e0758ff3ea2ca81799d06c0d18dc7d29e4d128a3ef4590c
```

All three stage protocols are explicitly blocked and non-operative. There are
zero C84 execution locks.

## Authorization Boundary

The PI directly expressed authorization intent for C84P/C84C/C84F/C84S. C84P
needs no scientific execution authorization. The future-stage intent was not
consumed because those stages have no unique executable lock and their required
prior gates have not passed. No token or verbatim hash recital is required, but
execution after repair still requires a server record binding a direct stage
authorization to the repaired, scope-specific protocol and lock.

## Validation

S0-S14 plus auxiliary max-T and Q1/Q2 checks passed 20/20. The design preserves
target-subject clustering for target counts 20, 22 and 76, same-method
cross-dataset intersections, panel/seed heterogeneity, budget closure and atomic
failure handling. The synthetic code imports no MOABB, MNE, PyTorch or training
adapter.

```text
focused C84P:   28 passed                         job 895316
C65-C84P:      514 passed, 1 skip, 3 deselected job 895317
C23-C84P:      921 passed, 1 skip, 3 deselected job 895318
full OACI:   1,849 passed, 1 skip, 3 deselected job 895319
final red team: 65 / 65 audit checks PASS
open scientific blockers: 1
```

All regression stderr is empty. No C84 path was skipped or deselected.

## Protected State And Next Step

```text
real EEG arrays loaded:          0
real labels read:                0
external dataset downloads:      0
training/forward/re-inference:   0
GPU jobs:                        0
candidate units created:         0
C84 execution locks:             0
C84C/F/S authorization consumed: 0
same-label oracle access:        0
BNCI2014_004 access:             0
manuscript drafting:             0
```

The next action is PM review of an additive availability-only common-montage
repair. Do not silently drop FCz, substitute Fz, interpolate, or start C84C/F/S
under the current objects.
