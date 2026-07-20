# C84L1P Overall Report

## Decision

C84L1P prospectively defines the previously missing fixed-zoo level-1
intervention and locks an engineering-only C84L1C adapter. It performs no real
EEG access, label read, training, forward pass, GPU allocation, target metric,
or scientific comparison.

Final gate:

```text
C84_LEVEL1_FIXED_PANEL_SUPPORT_DELETION_LOCKED_READY_FOR_ENGINEERING_CANARY_AUTHORIZATION
```

This gate authorizes nothing. A fresh direct `授权 C84L1C` statement is required
before the future runtime may consume the unique current lock. C84F and C84S
remain unauthorized and have no execution lock.

## Chronology And Identities

```text
C84FL blocked base:        6d6030f17dc2cdf8c8b180a9376632e238d42e75
C84L1 protocol commit:     a90f0051ed41937737ac7ac0258a882d45cefb33
C84L1 implementation:      4db7343868886d2cc05cbed18caa21092f2fe351
C84L1C lock commit:        3eafd70795344c43e0c6326e5c190ecaea4c2934
verification/report base:  12dc54fe0f853234b47913bd71f885b7b158414a

repair protocol SHA-256:   b9fed16afe9961d0d25f4801fa29859a8acb87c091e125ffe57e20e72ad00f35
external V3 SHA-256:       cd338bf1b97180b27dd42471dd2a67768194b74ed9896275424591d17e970ce0
canary V1 SHA-256:         88dad89054b4201be13ef5e6d63693bdfbc31a8646432f7e1a8c1d6ab094091b
field V5 SHA-256:          ce2bc1bb6a6c51e4ed1f4ba474c43fcc06638b2e01fafbf9f229af14904cfa94
science V3 SHA-256:        bf6c7f718413b4b2ac2ad9786aa2e47dc045a536e7237d5d8c0464b6598130b8
C84L1C lock SHA-256:       d6ccab97ebfbb1e1d571b71d5062e88dcfa08371ae9d53526cf7c25f45220e58
```

The C84FL blocker and all historical level-1 planned IDs remain preserved.
The repair protocol was committed before the level-1 implementation. No C84
level-1 real data or outcome existed before either object.

## Scientific Level Definitions

Level 0 remains `C84_LEVEL0_FULL_SOURCE_PANEL_V1`: the exact locked 12-subject
source-training panel with no deletion. All 972 level-0 unit IDs remain
unchanged, including the 243 accepted C84C panel-A/seed-5 units.

Level 1 is now
`C84_LEVEL1_FIXED_PANEL_LEFT_HAND_CELL_DELETION_V1`: before support-graph and
training-plan materialization, remove every source-training row for one fixed
subject and the canonical `left_hand` class. This is target-independent,
source-only, availability-blind after registration, paired to level 0, and not
an exact replication of C78's target-specific level.

| Dataset | Panel | Deleted source subject | Deleted class |
|---|---|---:|---|
| Lee2019_MI | A | 31 | left_hand |
| Lee2019_MI | B | 16 | left_hand |
| Cho2017 | A | 17 | left_hand |
| Cho2017 | B | 37 | left_hand |
| PhysionetMI | A | 103 | left_hand |
| PhysionetMI | B | 109 | left_hand |

The runtime permits no alternative cell. It requires the original 24 cells,
at least 8 rows in every cell, exactly 23 post-deletion cells, retained
right-hand support for the deleted subject, unique unchanged remaining trial
IDs, and unchanged source-audit and target rows.

## Paired Training And Candidate Identity

For each dataset/panel/seed pair, level 0 and level 1 use the same architecture,
optimizer, hyperparameters, epoch counts, cadence, base seed and model-init seed
rule. Plans are separately materialized from the level-specific population
signature. The future canary must replay each accepted C84C level-0 plan before
level-1 training and must prove equal level-0/level-1 model-init hashes.

```text
unchanged level-0 IDs:             972
superseded historical level-1 IDs: 972
new operative level-1 IDs:         972
complete operative IDs:          1,944 unique
C84L1C subset:                      243 units / 9 phases
```

New level-1 IDs bind the interface, montage, level intervention, deleted
subject/class and deletion-registry SHA-256. No historical planned level-1 ID
is operative.

## Accepted C84C Replay

The lock replays the accepted job `895441` complete-manifest SHA-256
`530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b`.
It binds 243 unchanged level-0 unit IDs, each dataset's four plan hashes, and
the model/checkpoint/optimizer/source-audit/sidecar registry digest
`0f455f9a605dc4427f9a8c10c1ff3e8fa0880bedbb383d283a165e6d3107b2cf`.
C84C target artifacts remain canary slices only and are not expanded here.

## Future C84L1C Scope

```text
datasets:       Lee2019_MI / Cho2017 / PhysionetMI
panel/seed:     A / 5
level:          1
targets:        19 / 24 / 106
units/phases:   243 / 9
role:           engineering only
```

The adapter must produce and replay 243 checkpoints, optimizer states,
sidecars, strict-source audit artifacts and canary-target unlabeled artifacts.
Target-y access and target scientific metrics are structurally forbidden. It
must not compare level-0 and level-1 target performance.

## Runtime Lock

The execution lock binds 39 implementation files and 107 runtime objects by
SHA-256 and Git blob, five protocol sidecars, the exact 20-channel montage,
environment/package/loader identities, deletion registry, candidate universe,
C84C plan/model registry, persisted-artifact checks and attempt ledger. Any
drift fails before authorization consumption, protected import or dataset
access.

The lock status is
`LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED`. The new authorization
record and content-addressed external output root are absent.

## Synthetic And Contract Validation

All 18 registered fail-closed fixtures passed. The expanded focused suite ran
163 tests, including historical C84 interface/runtime/result checks and all 36
new C84L1 intervention/protocol/canary tests. It covers wrong subject/class,
target/outcome-dependent choice, missing/low-support cells, protected-row
mutation, level-0 drift, unpaired initialization, historical-ID reuse, missing
intervention digest, target-y access and scientific-output emission.

## Regression Verification

| Suite | Slurm job | Passed | Skipped | Deselected | Stderr bytes |
|---|---:|---:|---:|---:|---:|
| focused | 895843 | 163 | 0 | 0 | 0 |
| C65 | 895844 | 649 | 1 | 3 | 0 |
| C23 | 895845 | 1060 | 1 | 3 | 0 |
| full | 895846 | 1984 | 1 | 3 | 0 |

All jobs ran CPU-only in `c84c-eeg2025-v3-exact` with 48 CPUs, 96 GiB and GPU
allocation 0. The single conditional skip is finalized C78F. The three
deselections are established C79 authorization-state tests and conceal no
C84L1 path. Every stderr file is empty.

## Red Team, Risks, And Hygiene

All 52 final red-team checks passed. All 19 registered risks are
closed and nonblocking. Git contains no raw EEG, weights, optimizer states,
NumPy caches or tracked file over 50 MiB. No C84L1 job remains active. The
shared branch was clean with `HEAD == origin/oaci` before report generation.

## Evidence Boundary And Next Stage

C84L1P establishes only a prospective source-support intervention and a locked
engineering implementation. It establishes no target performance, scientific
effect, external validity or level comparison. C84C level-0 reuse remains
valid. After a fresh C84L1C authorization and successful engineering review,
the next protocol milestone is C84FL2 for the remaining 1,458 units / 54 phases
and complete 76,464 target-context instrumentation slices. C84F and C84S still
require later, separate locks and authorizations.
