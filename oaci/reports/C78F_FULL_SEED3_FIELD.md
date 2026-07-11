# C78F — Full Seed-3 Multi-Regime Instrumented Field Generation

## Status

```text
Primary:   C78F-A_full_seed3_field_executed_and_manifested
Final gate: FULL_SEED3_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED
Protocol SHA-256: 85aba93fe2e232f0434162b3c6c97a30cac02047228676951c25cbab805d3d84
Authorization: direct explicit user authorization; no magic token
```

## Field

The locked remaining field completed exactly:

```text
remaining targets:             8
remaining target×level cells: 16
remaining training phases:    48
remaining retained units:  1,296
  ERM:                         16
  OACI:                       640
  SRC:                        640

complete seed-3 field:      1,458 units
  ERM:                         18
  OACI:                       720
  SRC:                        720
strict-source rows:       6,718,464
target-unlabeled rows:      839,808
```

Target 4 remains the previously observed engineering canary and is excluded from
all C78S primary tests. The remaining eight targets are the locked seed-3
exploratory replication field, not an independent target-population confirmation.

## Waves

`Wave A targets=[8, 9, 3, 6] units=648 engineering=1; Wave B targets=[5, 2, 7, 1] units=648 engineering=1`

Wave B was released only after Wave A passed checkpoint, hash, target-isolation,
instrumentation, physical-view, storage, and numerical-identity gates. No target
scientific outcome or label was used for continuation.

## Integrity

```text
checkpoint/state/sidecar hashes: 1,458 / 1,458 pass
new optimizer-state replays:     1,296 / 1,296 pass
cadence cells:                       54 / 54 pass
new genealogy rows:              1,296 / 1,296 pass
target training rows/labels:         0 / 0
source-audit training rows:              0
Wz+ b/logit max abs:             0.000e+00
softmax max abs:                 0.000e+00
hook-z max abs:                  0.000e+00
identity failures:                       0
```

Strict-source, target-unlabeled, construction, evaluation, and same-label oracle
views are physically separated. Label views were materialized only after the
complete 1,458-unit field was frozen. The generation and primary instrumentation
paths never received a label-view or oracle descriptor.

## Resources

```text
remaining measured GPU phase-wall sum: 6.862118 h
remaining measured external payload:   26,766,911,921 bytes
instrumentation job-wall sum:           1417.922 s
```

These are measured values. GPU phase-wall sums are not presented as elapsed
calendar time because targets within a wave ran concurrently.

## Engineering Repair

Collector job `893052` failed after field generation on a compact-schema key
mismatch: the frozen `c74_cache` descriptor exposes `row_count`, while the
collector requested `rows`. Training, instrumentation, field freeze, and label
isolation had already passed; the failure performed no training, forward pass,
GPU work, target-label read, or target-metric computation.

The repair was additive and prospectively locked at commit `f0d49c2` (protocol
SHA-256 `60c4c4f2a9a78e7d68af995a6319989c0d9cb46f331d9af54260f3d7e76b508e`).
Its independent red team passed 10/10. Replacement job `893055` changed only the
compact descriptor mapping, retained the failed job in the repair ledger, and
left every execution-locked training/instrumentation file byte-identical.

## Boundaries

C78F computed no target accuracy, calibration, transport, association,
actionability, or checkpoint-selection result. It creates no selector and emits
no checkpoint recommendation. SRC remains the historical negative control; ERM
remains an anchor rather than a symmetric trajectory.

C78S is hash-locked and ready but has not started. Seed 4 and BNCI2014_004 remain
untouched. C79 remains unauthorized.

## Red Team

Independent pre-report red team: 57/57 blocking checks pass.
The authorization simplification is explicit: direct user approval is bound to
the committed protocol scope through the execution lock, with no token ceremony.

## Regression

- `focused` job `893056`: 32 passed, 0 failed
- `c65_c78f` job `893057`: 214 passed, 0 failed
- `c23_c78f` job `893058`: 621 passed, 0 failed
- `full_oaci` job `893059`: 1549 passed, 0 failed
