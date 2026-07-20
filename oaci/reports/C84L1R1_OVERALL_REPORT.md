# C84L1R1 Overall Report

## Decision

The first authorized C84L1C engineering attempt stopped safely on a numerical
identity gate. C84L1R1 preserves that failure, calibrates a narrowly scoped
float32 reconstruction tolerance, provides a replacement implementation and
locks a complete fresh 243-unit engineering canary.

Final gate:

```text
C84L1C_FLOAT32_REPLAY_REPAIRED_AND_RELOCKED_READY_FOR_FRESH_PI_AUTHORIZATION
```

This is a readiness gate only. The original authorization was consumed. No
replacement C84L1C execution, C84F execution or C84S execution is authorized.

## Chronology And Identities

```text
C84L1P readiness HEAD:      a0ec77b3a41084106713bf1f259e1daad2004607
authorization commit:       05bfca18c58b67b6cc0b7c5d57dfc7dc1036f8ea
failed Slurm job:            895928
repair protocol commit:      e35ba0bfb412fbdcbc6fb127db05af1d91f51440
replacement implementation: d0159d1b2db26d796ae3f9853329a5851aa93222
replacement lock commit:     afc5a6b5aedbb0e9d9b09acba0997657513e5268
regression base:             d7ba0c69b193736ab6667e0272d9287a171683e3

repair protocol SHA-256:     2e199f6f63dffd1b02c1e31102ed189e31bf6e4961465394230f8e9de1d4ddf0
canary V2 SHA-256:           6e6bcb6b60726c76c8db0afc48e954d0e4a1cf68bfd29796987bfd6828355616
field V6 SHA-256:            cd8646403a7564e9d1a7e3d64104483cbd56ac85bebaafa1244afdde8a8ed310
execution lock SHA-256:      f9ebd88c72915bb41ba2d2d84a2a00c6748272021d48043c299bce52a1ad3813
```

The external V3 and science V3 contracts remain unchanged. The replacement
lock also binds their hashes, the accepted C84C level-0 evidence, the exact
20-channel montage and the fixed level-1 support-deletion registry.

## Failed Engineering Attempt

Job `895928` ran on `node08` with one V100 GPU, eight CPUs and 64 GiB. It
consumed authorization, entered Lee2019_MI, completed three training phases and
materialized all 81 optimizer states. It froze 73 complete units and failed
while instrumenting the next SRC unit at epoch 164 / trajectory order 33.

```text
failed unit:                 c84l1_3694c9bcfa6865ae595f4607e9c03a0f
observed linear error:       1.239776611328125e-5
locked linear tolerance:     1e-5
softmax error:               0
repeat-logit error:          0
repeat-z error:              0
elapsed from attempt ledger: 2041.9867 seconds
```

The partial state contains 74 checkpoints, 81 optimizer states, and 73 each of
sidecars, strict-source artifacts and target-unlabeled artifacts. Its manifest
SHA-256 is
`ba67a4a0f8a516085b3eb020c353c401c2eafdd1981eb880c5c63587ac31b091`.
The failed root is preserved and cannot be reused.

Protected counters are all zero:

```text
target-y accesses:                 0
construction-view access:          0
evaluation-view access:            0
same-label-oracle access:           0
target scientific metrics:         0
target-outcome decisions:           0
```

This was a real engineering execution, not a no-data dry run. It produced no
target-label or scientific result and creates no level-1 performance claim.

## Numerical Repair

The 73 persisted engineering artifacts were inspected only for numerical
identity magnitudes. Their maximum persisted `Wz+b` difference was
`7.62939453125e-6`; the failing in-memory unit reached
`1.239776611328125e-5`. No label or scientific endpoint entered this
calibration.

C84L1R1 changes only:

```text
float32 zW+b reconstruction: 1e-5 -> 2e-5
```

The scope is the 1040-term CPU/GPU float32 classifier reconstruction. Saved
softmax, repeat logits and repeat z remain at `1e-6`. Nonfinite values, unknown
fields and errors above `2e-5` fail closed. No model, optimizer, RNG, training
plan, data view, subject, candidate ID, intervention or scientific rule changes.

## Replacement Canary Contract

The scope remains panel A / seed 5 / level 1 for Lee2019_MI, Cho2017 and
PhysionetMI, with targets 19 / 24 / 106, 243 units and nine training phases.
The exact fixed deletion cells remain subject 31 / 17 / 103 times
`left_hand`.

The replacement must use a fresh content-addressed root and retrain all 243
units. None of the 73 complete or other partial artifacts from job `895928` may
be retained. Completion still requires 243 checkpoints, optimizer states,
sidecars, strict-source artifacts and target-unlabeled artifacts, plus all
support, plan, persisted-artifact and paired-initialization replays.

## Runtime Lock And Authorization

The lock binds 44 implementation files and 125 runtime objects by bytes,
SHA-256 and Git blob, plus five protocol identities. It replays all objects,
the exact environment and loaders, C84C level-0 evidence, candidate digest and
intervention registry before authorization consumption or data access.

The replacement authorization record
`oaci/reports/C84L1C_PI_AUTHORIZATION_RECORD_V2.json` is absent, and the
replacement root
`/projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v2` does not
exist. A manual preflight completed all 125 object and five protocol replays,
then failed closed at the missing authorization without creating the root.

## Validation And Regression

All 12 synthetic/contract cases passed, including the observed error, a value
above `2e-5`, unchanged strict checks, persisted replay, nonfinite/unknown-field
failure, fresh paths, unchanged scope, failed-root rejection and pre-root
authorization failure.

| Suite | Slurm job | Passed | Skipped | Deselected | Stderr bytes |
|---|---:|---:|---:|---:|---:|
| focused | 896000 | 175 | 0 | 0 | 0 |
| C65 | 896001 | 661 | 1 | 3 | 0 |
| C23 | 896002 | 1,072 | 1 | 3 | 0 |
| full | 896004 | 1,996 | 1 | 3 | 0 |

All used `c84c-eeg2025-v3-exact`, 48 CPUs, 96 GiB and GPU allocation 0. The
single skip is finalized C78F; the three deselections are historical C79
authorization-state tests. `squeue` shows all listed jobs inactive.

## Red Team And Hygiene

All 44 red-team checks and all 19 risk controls pass or are accepted with
explicit disclosure. No tracked raw EEG, weight, optimizer, cache or NumPy
payload exists, and no file under `oaci/` exceeds 50 MiB. The failed attempt,
consumed authorization and failed root remain preserved in history.

## Evidence Boundary And Next Action

C84L1R1 is an engineering numerical-replay repair. It does not establish target
performance, a level effect, external validity or any scientific endpoint.
C84F and C84S remain unlocked and unauthorized.

The next permissible action is a new direct `授权 C84L1C`. The server must bind
that statement to canary V2 and execution lock V2, create a new authorization
record, and run the complete 243-unit replacement in the fresh root. The prior
authorization cannot migrate.
