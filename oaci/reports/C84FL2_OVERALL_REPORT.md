# C84FL2 Overall Report

## Decision

C84FL2 reconciles the accepted level-0 and level-1 engineering canaries into a
single fail-closed dual-level C84F implementation. It creates the one permitted
C84F execution lock and stops before authorization or real execution.

Final gate:

```text
C84F_DUAL_LEVEL_FULL_FIELD_IMPLEMENTATION_AND_EXECUTION_LOCK_READY_FOR_PI_AUTHORIZATION
```

This is a readiness gate only. C84F remains unauthorized and unexecuted. C84S
has no execution lock and is not authorized.

## Chronology And Immutable Identities

```text
accepted C84L1C base:       4d2ca75b2fc149c80c3e51e93709aab12e67813a
C84FL2 protocol commit:     24a21d795ccfaa8cfd816f77eaa9f41867fad847
C84FL2 implementation:      96af9f3751451f7def6d9b35b8ab395675e41394
C84F lock commit:           47bf20fbc341c136da0e3ed997a490fb0f135c49
regression replacement:     196bb44b334c5f41d334988a00714b7c3e85c3f0
verification/report base:   2a957a264293e119f5e5b0c8d89bcf3410eef755

reconciliation SHA-256:     2ac679a5308d5d972b14d38e01cbc0d875ca6c5e547b752945fc831e38081f62
field V7 SHA-256:           9db0219befecb11cf72386b96e28ee9d9430c3df5d7947298f102492f072b737
full-field V2 SHA-256:      dafc44dbc24ea5d4d1cea61207479cbd986c9f8129b111682a00f15a44b1d15d
C84F lock SHA-256:          f9df9dcefea59b05bfea24d1b744d82bfc933d76efde3f9aececf67401ea6b05
operative registry SHA-256: 462fa840f7048511cb3e1a41b55f60441435d14f665014b856fe5fd8d66ac1b0
```

All older C84FL and field protocols remain historical and unchanged. The
additive protocol commit precedes implementation. C84FL2 accessed zero real EEG,
labels, training, forward, GPU, selector outcomes or scientific endpoints.

## Accepted Dual-Canary Reuse

The lock byte/hash-replays both accepted engineering fields:

```text
C84C level 0:   243 units / 9 phases
C84L1C level 1: 243 units / 9 phases
combined:       486 units / 18 phases
replayed external artifact files: 2,430
```

Reusable objects are candidate identity, checkpoint, optimizer state, sidecar,
genealogy/state descriptor and strict-source audit artifact. Failed jobs
`895366` and `895928` and their roots are explicitly rejected. The six canary
target contexts and 486 candidate-context slices are replay witnesses only;
they are not substituted for the final uniform target field.

## Complete Field And Remaining Waves

The operative field has 1,944 unique candidate units: 972 level 0 and 972 level
1, across 24 zoos and 72 phases. Level 0 keeps the full 12-subject source panel.
Level 1 uses the fixed registered subject x `left_hand` deletion. Each paired
dataset/panel/seed cell uses equal model initialization and otherwise identical
architecture, optimizer, epochs, cadence and deterministic settings.

Remaining work is exactly 1,458 units / 54 phases:

| Wave | Scope | Units | Phases |
|---|---|---:|---:|
| A | panel A / seed 6 / both levels | 486 | 18 |
| B0 | panel B / seed 5 / both levels | 486 | 18 |
| B1 | panel B / seed 6 / both levels | 486 | 18 |

Wave release can inspect engineering identity and replay only. It cannot inspect
target arrays, predictions, calibration, accuracy, selector scores, regret,
Q1/Q2 or label-budget outcomes.

## Model Freeze And Target Barrier

No new target subject may be loaded until an atomic model-field manifest proves
1,944/1,944 checkpoints, optimizer states, sidecars and strict-source artifacts,
72/72 phases, 486/486 reused units, 1,458/1,458 new units, 972/972 units per
level, and zero target rows/labels or outcome-driven retention/retry.

Only after that barrier may the unlabeled interface create a label-free trial
registry for all 118 target subjects. The structural target-y slot is never
indexed, represented, converted, hashed, summarized or logged.

## Complete Target Instrumentation

The locked target stage creates one uniform all-target artifact per unit:

```text
unit artifacts:             1,944 / 1,944
target contexts:              944 / 944
candidate-context slices:  76,464 / 76,464
target subjects: Lee 22, Cho 20, Physionet 76
```

The target module has no training callable. Instrumentation failure cannot
retrain or alter model retention. All six canary contexts must match by trial
and candidate ID and replay logits, probabilities and z. The field-wide linear
gate is fixed at `2e-5`; softmax, repeated logits and repeated z remain `1e-6`.
No tolerance can be widened at runtime.

## Runtime Lock

The lock replays 79 repository objects, 38 implementation
files, seven protocol sidecars, the exact environment and loader sources, the
20-channel interface, all 1,944 IDs and all 2,430 reusable external artifacts.
It requires clean `HEAD == origin/oaci`, a fresh direct C84F authorization and
an empty content-addressed output root before protected imports or data access.

Lock status:

```text
LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED
```

No authorization record exists. The shortest future direct statement is
`授权 C84F`; it must be server-bound to this unique lock and protocols.

## Resource Envelope

Measured dual-canary training was 10,734.285 seconds over 18 phases. The linear
remaining projection is 8.945 GPU-hours; the complete 5x safety estimate is
59.635 GPU-hours, below the 250-hour ceiling. Download plus derived payload is
projected at 245,201,501,528 bytes, below 2 TiB. The complete target
instrumentation projection is 49,036,391,984 bytes. No runtime scope reduction
is allowed.

## Synthetic Validation And Regression Lifecycle

All 25 synthetic/contract fixtures passed, including dual-canary identity,
failed-root rejection, unit/wave arithmetic, paired initialization, target
barrier, target-y failure, full-context coverage, canary replay, numerical
gates, atomic manifests and authorization fail-closed behavior.

The first regression attempt correctly exposed six stale historical tests that
still asserted no C84F lock could exist:

| Suite | Failed job | Passed | Failed | Replacement job |
|---|---:|---:|---:|---:|
| focused | 896157 | 221 | 6 | 896163 |
| C65 | 896158 | 707 | 6 | 896164 |
| C23 | 896159 | 1118 | 6 | 896165 |
| full | 896160 | 2042 | 6 | 896166 |

Those tests were repaired to assert the historical no-lock state at the
historical commit and the current not-authorized/unexecuted lock at the current
commit. No runtime or scientific object changed.

| Suite | Slurm job | Passed | Skipped | Deselected | Stderr bytes |
|---|---:|---:|---:|---:|---:|
| focused | 896163 | 227 | 0 | 0 | 0 |
| C65 | 896164 | 713 | 1 | 3 | 0 |
| C23 | 896165 | 1124 | 1 | 3 | 0 |
| full | 896166 | 2048 | 1 | 3 | 0 |

Every replacement ran on `cpu-high` with 48 CPUs, 96 GiB and GPU 0 in
`c84c-eeg2025-v3-exact`. The one cumulative skip is finalized C78F. The three
cumulative deselections are established C79 authorization-state tests. All
stderr files are empty. The leading-numeric parser includes C34S and every new
C84FL2 test in focused, C65, C23 and full as applicable.

## Red Team, Risks, And Hygiene

All 66/66 blocking red-team checks passed. All 30
registered risks are closed and nonblocking; the failure ledger is clear. Git
contains no raw EEG, model weights, optimizer states, NumPy caches or tracked
file over 50 MiB. No C84FL2 job remains active at report generation.

## Evidence Boundary And Next Stage

C84FL2 establishes implementation and lock readiness only. It provides no new
EEG evidence, target prediction, selector result, external-validity result or
scientific taxonomy. A future fresh direct C84F authorization may execute only
the locked field. Successful C84F must stop at
`C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED`. C84S requires a separate implementation, lock, PM
review and authorization after the complete field freezes.
