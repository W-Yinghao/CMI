# S2P_34 - Phase D1 Design Red-Team

**Status:** METADATA PREFLIGHT IMPLEMENTED / TRAINING NOT AUTHORIZED.

This is an adversarial protocol review. It does not launch or approve compute.

## Preflight result

The independent verifier passed all 13 frozen checks with no blockers:

```text
subset seed 0:
  U200 = 24,000 windows / 260 subjects / 392 recordings
  U1000 = 120,000 windows / 918 subjects / 1,882 recordings
  exact extension = 96,000 windows

subset seed 1:
  U200 = 24,000 windows / 299 subjects / 434 recordings
  U1000 = 120,000 windows / 967 subjects / 1,957 recordings
  exact extension = 96,000 windows

all six group quotas:
  exact for both U arms and both subset seeds

initial-state U-arm pairing:
  4/4 blocks exact

protocol snapshots:
  8 trajectories / 16 fixed-update snapshots
```

The two corpus replicates are materially different: their U200 arms share 673 windows and their U1000 arms
share 25,814 windows. This is expected and recorded; each subset is reused unchanged across both initialization
seeds.

Observed Route-B throughput gives an estimated training cost of `103.624--168.772 GPU-hours`, excluding queue
time and downstream audits. Sixteen primary payloads require approximately 0.95 GB before logs and optional
secondary checkpoints. Compute approval remains false.

## Fail-closed checks

| Risk | Required evidence | Disposition |
| --- | --- | --- |
| Budget subsets are not nested | Exact window identity inclusion for both subset seeds | Binding launch gate |
| Group composition changes with U | Exact 200h and 1000h quota replay in all six groups | Binding launch gate |
| Data and initialization are coupled | Full `subset_seed x init_seed` crossing | Binding launch gate |
| U arms start from different weights | Tensor-canonical initial-state SHA equality | Binding launch gate |
| Shuffle/mask RNG reuses init seed or each other | Separate stream, loader-root, and mask-root fields | Binding launch gate |
| P_low and P_high use different LR histories | One 93,750-step high-horizon schedule | Binding launch gate |
| More validation points create best-val advantage | Fixed-update primary snapshots | Binding launch gate |
| P_low is overwritten during continuation | Immediate content-addressed closure | Binding launch gate |
| Epoch-based behavior reappears | Global-update validation and snapshot assertions | Binding launch gate |
| Small-data arm is rescued ad hoc | No early stopping or result-visible rescue | Binding launch gate |
| Target labels affect pretraining choices | Target-label firewall | Binding launch gate |

## Scientific limits that remain after a pass

The factorial does not isolate every latent factor:

```text
U includes unique windows, subject coverage, recording breadth, and population breadth.
P includes optimizer updates, presentations, repeated exposure, and mask draws.
P_high shares trajectory history with P_low by design.
Only two subset seeds and two initialization seeds are available.
```

Consequently, legal terms are `unique-data breadth effect`, `cumulative training exposure effect`, and
`PILOT_FACTORIAL_ESTIMATE` where training-level uncertainty is weak. `Pure subject-count effect`, `pure optimizer
effect`, and a general scaling law remain forbidden.

## Infrastructure risks

The old trainer cannot be reused unchanged because it couples subset and initialization seeds, validates once per
epoch, compresses cosine LR to the requested epoch horizon, and retains mutable best/last checkpoint paths. A D1
launcher must therefore be reviewed against the frozen metadata package before any `sbatch` call. Passing this
red-team does not imply that such a launcher already exists.

The compute estimate must use observed Route-B throughput and include eight 93,750-update trajectories plus 16
immutable snapshots. Queue time and downstream audit cost are separate. GPU type or partition changes may affect
wall time but cannot alter batch size, accumulation, trajectory, RNG, validation, or snapshot contracts.

## Stop rules

Stop before or during training if any of the following occurs:

```text
1. U200 is not an exact window subset of U1000.
2. Init seeds receive different manifests within a subset/U cell.
3. U-arm initial-state hashes differ inside a block.
4. Batch size or gradient accumulation differs across arms.
5. LR is not the common 93,750-step schedule.
6. A fixed-update snapshot is absent, mutable, overwritten, or fails strict reload.
7. Update or presentation counts differ from the frozen table.
8. Validation is triggered by epoch rather than the global update grid.
9. Early stopping occurs.
10. Target labels enter checkpoint or protocol selection.
11. A content-addressed snapshot hash changes.
12. Any arm needs a result-visible rescue.
```

## Authorization boundary

The independent verifier writes a machine-readable red-team verdict. Even on PASS, the only valid state is
`PASS_PROTOCOL_ONLY_TRAINING_HELD`. Training requires a subsequent explicit PM decision.
