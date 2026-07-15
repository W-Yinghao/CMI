# S2P_38 - Phase D1 Implementation and Launch Contract

**Status:** IMPLEMENTATION CANARY AUTHORIZED / TRAINING AUTHORIZED ONLY AFTER CANARY PASS.

This is a technical experiment contract, not manuscript text. It does not authorize downstream analysis,
fine-tuning, H4000, CodeBrain, a new dataset, or submission writing.

## Scientific trajectories

The frozen D1 design remains unchanged:

```text
unique data: 200h, 1000h
subset seeds: 0, 1
initialization seeds: 0, 1
high-horizon trajectories: 8
primary snapshots per trajectory: updates 18,750 and 93,750
```

Each low-exposure snapshot is taken from the same in-memory trajectory that continues to the high-exposure
snapshot. The common cosine schedule has `T_max=93,750`; validation runs every 1,875 updates and has no checkpoint
selection role. Existing B1 checkpoints are historical anchors and are not D1 cells.

## Runner invariants

`route_b_phase_d1_train.py` fails closed unless all of the following hold:

1. the row-level manifest hashes to the preregistered window-identity hash;
2. the instantiated model hashes to the paired U-arm initial-state hash;
3. batches are full and have native shape `[64,33,30,200]`;
4. masks and stochastic forward state use an update-indexed, domain-separated stream seed;
5. optimizer and scheduler advance exactly once per global update;
6. primary snapshots occur only at the two fixed updates;
7. each snapshot is content-addressed, no-overwrite, mode `0444`, and reloads model, optimizer, and scheduler state;
8. the in-memory and reloaded models produce bitwise-identical unlabeled validation features;
9. no target labels, early stopping, or best-validation selection enter training.

## Paired implementation canary

The canary runs `SS0_IS0_U200` and `SS0_IS0_U1000` for four updates, with snapshots at updates 2 and 4 and
validation every two updates. It invokes the real data loader, model, native masked-reconstruction objective,
optimizer, scheduler, immutable closure, strict reload, and feature canary. It is an implementation test, not a
scientific D1 cell.

The verifier requires:

```text
same initial-state hash across U arms
same stream contract across U arms
same LR at matched updates
both fixed snapshots present for both arms
all payload hashes stable and read-only
model/optimizer/scheduler reload exact
feature reload and repeat max_abs_diff = 0
no downstream or target labels
```

Only a persisted `PASS` verdict authorizes submission of the fixed eight-task array. After array completion, only
the pretraining provenance gate is allowed; D1 downstream remains held until the FMScope-FSR Panel 2 closes.

## Resource boundary

Both 200h and 1000h D1 trajectories may use one GPU from the approved scheduler pool:

```text
A100, H100, A40, L40S, V100, V100-32GB, V100-16GB
```

The launcher must not change batch size, gradient accumulation, update horizon, partition policy, or split the
trajectory after seeing queue or runtime behavior.
