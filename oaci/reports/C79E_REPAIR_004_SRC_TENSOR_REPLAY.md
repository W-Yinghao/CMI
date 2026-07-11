# C79E Repair 004 - SRC Source Tensor Replay Affinity

## Failure

Wave-A SRC jobs `893466` (target 9) and `893463` (target 8) stopped
before optimizer execution while replaying their frozen ERM anchors. Every
registered context field passed except the bitwise source-population
`tensor_hash`.

```text
target rows read:                 0
target labels read:               0
model-specific outcomes read:     0
optimizer steps:                  0
SRC checkpoints created:          0
scientific registry changed:  false
```

The frozen ERM and failed SRC jobs ran on different Slurm nodes:

```text
target 9: ERM/OACI node43, failed SRC node13
target 8: ERM/OACI node14, failed SRC node43
```

The failure is an engineering replay failure, not permission to weaken the
exact tensor gate.

## Prospective replacement

Retry each failed SRC phase with the exact locked command, environment,
checkpoint universe, seed, target, levels, and objective on the node that
generated its frozen ERM anchor:

```text
target 9 replacement: node43
target 8 replacement: node14
```

No source tensor, checkpoint, sidecar, model, threshold, null, scientific
registry entry, or execution lock is changed. The existing exact
`tensor_hash` comparison remains mandatory. The replacement is accepted only
if it passes that unchanged gate and all downstream SRC identities. Failed and
dependency-cancelled jobs remain in the attempt ledger.

