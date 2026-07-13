# C80R Implementation Replay

## Scope

C80R is an additive, pre-outcome repair of the C80 existing-field analysis
boundary. It does not execute C80E and does not consume any budget-specific
scientific outcome. The historical protocol `f5d83b3`, lock `972f47c`, blocked
authorization record, and preflight evidence `6c18fd4` remain unchanged in Git
history.

## Chronology

```text
6c18fd4  accepted safe preflight stop; outcome reads = 0
e88a244  additive repair protocol and taxonomy lock
0e9dedc  nested authorization guard plus fail-closed real adapter
9617760  first replacement lock revision
e5cb41a  preauthorization completeness refinement
f19acd8  final replacement analysis lock
```

The `9617760` lock revision is preserved and explicitly superseded. The
refinement was based only on protocol/report-schema red-team findings; it added
source-relative top-k quantities, target-cluster secondary bands, paired
cross-seed rows at all budgets, and S3 leave-one-target-out reporting. It did
not change the 80/80 scientific registry, estimands, budget grid, Q0 policy,
selector, RNG streams, Monte Carlo count, materiality thresholds, dependence
units, or taxonomy.

## Adapter Boundary

The operative adapter is
`oaci/conditioned_ceiling_coverage/c80r_existing_field_adapter.py`, SHA-256
`7e5ac0ba829bf5f233ed469f6fb8f6da4054d0bf4d024a0736a45e3674f1b56c`.
It is bound by lock commit `f19acd8`, lock SHA-256
`e18f2b5f1d79b6fcd96207339c5842e30b7aecb5bc22b8939a475487068b1b82`.

The runtime is fail-closed in this order:

```text
replacement protocol/hash replay
-> implementation and manifest replay
-> new direct PI authorization replay
-> construction-only nested-Q0 selection
-> content-addressed selection freeze
-> selection-manifest hash replay
-> physically separate evaluation opening
-> unconditional P1/P2/S1/S2/S3 execution
-> machine-readable result freeze
-> later narrative/red-team stages
```

The canonical authorization field is `lock.protocol.sha256`. The old
authorization is rejected explicitly and the new record path is distinct:
`oaci/reports/C80E_REPAIRED_PI_AUTHORIZATION_RECORD.json`.

## Scientific Identity

The repair adds only the PM-locked decision taxonomy:

```text
1 C80-E: blocker
2 C80-D: either seed B* absent
3 C80-B: both B* exist but stability fails
4 C80-C: stability passes and both B* are in {32,FULL}
5 C80-A: stability passes and C80-C is false
```

`FULL` remains the complete construction view within each exact cell. It is not
the number 61 and is not numerically interpolated with budget 32.

## Protected State

At C80R lock time:

```text
real budget statistics:        0
evaluation-label value reads:  0
same-label oracle accesses:    0
target4 primary rows:          0
training/forward/re-inference: 0
GPU jobs:                      0
new C80E authorization:        absent
```

Only committed compact files, external JSON manifests, route metadata, and
synthetic/schema fixtures were read. No external NPZ payload was opened by
C80R.

