# C78 — Seed-3 OACI+ERM Instrumented Training Pilot / Full-Field Expansion Gate

**Final gate:** `PILOT_READY_BUT_NOT_AUTHORIZED`

**Primary execution taxonomy:** `not evaluable; no P1 execution occurred`

**Secondary active:** `C78-S8 + C78-S9 + C78-S11`

## Gate-first result

```text
planned field:          82 units
ERM anchors:             2
OACI trajectory units:  80
SRC units:                0
training attempted:       0
real EEG forward:         0
real EEG rows loaded:     0
GPU requested:            0
checkpoints created:      0
raw cache rows:           0
seed-4 access:            0
BNCI2014_004 access:      0
```

The exact CLI authorization token was not passed to the C78 command. Prompt text, generic approval language, environment variables, whitespace variants, and substring matches were not accepted. This is therefore the required no-training P0 result, not a failed training run.

## Protocol and scope

- C78 protocol anchor: `23f549d`.
- Full protocol SHA-256: `ad6f4e034318b879755ca46a719d39cfd3d3c36d7ee8478771d08778a8b71afc`.
- Accepted C77 result: `285ba1d`; the protocol anchor is its prospective ancestor.
- Explicit unit manifest: `82/82` unique planned units, target `4`, seed `3`, levels `0 + 1`.
- Per level: one shared ERM stage-1 final anchor and OACI epochs `4,9,...,199` (`40` fixed-cadence records).
- The protocol's 1,458-unit `execution_matrix` was not treated as C78 authorization.

Seven historical code/config identities replay byte-exact, including ERM, OACI, the training engine, and the confirmatory manifest. ERM and OACI remain asymmetric: ERM is a shared anchor; OACI is the trajectory.

## P0 readiness

```text
locked environment SHA match: 1
storage free snapshot:         2791.370 GiB
required temporary reserve:    3.013 GiB
storage capacity pass:         1
dummy Wz+b max error:          0.000e+00
dummy softmax max error:       0.000e+00
dummy repeat logit/z error:    0.000e+00 / 0.000e+00
```

The dummy ABI used CPU synthetic inputs only. Real Wz/logit identity, training determinism, target isolation, checkpoint genealogy, cadence completeness, runtime, and cache materialization remain explicit P1 runtime gates; their tables report zero checked real units rather than inheriting the dummy pass.

## Isolation boundary

Six physically separate view schemas are locked. The training process receives source training inputs only; target-unlabeled instrumentation and target label views are deferred until all 82 retention decisions and checkpoint manifests are frozen. The same-label-oracle path is unavailable to primary pilot validation. These are execution contracts, not claims that runtime isolation has already passed.

## Red team

Independent red team passed `67/67` blocking checks before this report was created. Its principal repairs were:

- `R1_protocol_parent_semantics`: replay now requires the anchor to be an ancestor of accepted C77 result 285ba1d; it correctly retains C76 as its lock-time parent
- `R2_authorization_phrase`: generic prompt text was rejected; training/forward/GPU/data counters remain zero
- `R3_execution_taxonomy`: all primary execution taxonomy remains not evaluable; only readiness gate and boundary secondaries are reported
- `R4_runtime_identity`: dummy and real identity tables are separate; real rows/units checked remain explicitly zero
- `R5_SRC_coverage`: full seed-3 field remains blocked behind prospective SRC canary or exact-path proof and new PM review
- `R6_power_materiality`: C78 makes no H2/materiality claim and requires future power re-lock
- `R7_partition_row_normalization`: partition checks now operate only on rows with a non-empty partition; job 892802 is retained as the blocking failed attempt

Regression: focused_C78 12 green (job 892811), C65_C78 150 green (job 892812), C23_C78 557 green (job 892813), full_OACI 1485 green (job 892814).

## Decision

C78 is ready for a separately invoked exact-token P1, but it has not trained or instrumented the 82-unit field. Consequently none of `C78-A` through `C78-E` is active. No measurement-control replication, cross-regime transport result, representation mechanism, strict-source escape hatch, selector, checkpoint recommendation, deployability, or target-population claim is made.

Even a future successful OACI+ERM P1 cannot authorize the 1,458-unit expansion. SRC was not exercised, so PM review must first choose a prospective SRC canary or prove that SRC shares the exact validated execution/instrumentation path.
