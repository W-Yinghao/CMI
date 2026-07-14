# S2P_32 - Phase D0 Schedule and Presentation Audit

**Status:** D0 COMPLETE / INDEPENDENT VERIFIER PASS / D1 HELD FOR PM REVIEW.

This is a technical experiment record, not a manuscript, abstract, or submission narrative. Phase D0 reads
existing Route-B logs, immutable checkpoints, and corpus manifests. It does not launch pretraining,
fine-tuning, CodeBrain work, H4000, layerwise analysis, or a new downstream audit.

## Question

Phase D0 determines whether the existing budget comparison separates unique EEG data from repeated
optimization. It reconstructs, for every Route-B run:

```text
unique windows
completed epochs
optimizer updates
actual sample presentations
effective batch size
gradient accumulation
learning-rate schedule length
validation/checkpoint cadence
val-selected checkpoint epoch
```

The audit also checks exact window identity across budgets and initialization seeds. A subject-count hash or
nominal hour total is not sufficient to establish nested data.

## D0 findings

All eight Route-B runs completed 50 epochs with batch size 64, no gradient accumulation, no window cap, and the
same byte-identical trainer and loader sources. The reconstructed schedules are:

| Budget | Unique windows | Steps/epoch | Total updates | Actual presentations | Presentation-hours equivalent |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200h | 24,000 | 375 | 18,750 | 1,200,000 | 10,000h |
| 500h | 60,000 | 937 | 46,850 | 2,998,400 | 24,986.67h |
| 1000h | 120,000 | 1,875 | 93,750 | 6,000,000 | 50,000h |
| 2000h | 240,000 | 3,750 | 187,500 | 12,000,000 | 100,000h |

H500 drops 32 windows per epoch because 60,000 is not divisible by 64. The other three budgets have no tail.
The existing H200-to-H1000 contrast therefore multiplies unique data, optimizer updates, and sample
presentations by five simultaneously. It cannot identify a unique-data effect.

The exact-window reconstruction also found that the existing budget subsets are not nested. Only 5,125 of the
24,000 H200_s0 windows (21.35%) occur in H1000_s0; for seed 1, the overlap is 4,640 windows (19.33%). In
addition, `subset_seed == init_seed`, so the two initialization seeds use different EEG subsets. The fixed
subject-disjoint pretrain-validation pool is identical across all eight runs.

Checkpoint selection is pretrain-validation-loss-only, but validation currently occurs once per epoch. This
means H200 and H1000 checkpoints have different update intervals and different selection opportunity sets.
The selected checkpoints reached 46--50 epochs, so selected-state presentations also differ slightly from the
full planned schedules.

## Frozen arithmetic

The existing trainer uses one optimizer update per full 64-window batch, drops the final incomplete training
batch, and advances `CosineAnnealingLR` once per optimizer update. There is no gradient accumulation. Therefore:

```text
steps_per_epoch = floor(unique_windows / 64)
total_updates = steps_per_epoch * completed_epochs
actual_presentations = total_updates * 64
```

Validation uses all validation windows and currently runs once per epoch. The selected checkpoint is the lowest
pretrain-validation loss among those epoch checkpoints. Phase D0 reports both the full scheduled exposure and
the exposure reached by the selected checkpoint.

## Prospective D1 map

If D1 is later approved, the minimal exact presentation-matched factorial is:

| Cell | Unique data | Epochs | Optimizer updates | Sample presentations |
| --- | ---: | ---: | ---: | ---: |
| U200-P_low | 200h | 50 | 18,750 | 1,200,000 |
| U1000-P_low | 1000h | 10 | 18,750 | 1,200,000 |
| U200-P_high | 200h | 250 | 93,750 | 6,000,000 |
| U1000-P_high | 1000h | 50 | 93,750 | 6,000,000 |

The LR scheduler must use the matched update total as `T_max`. Validation and checkpoint selection must also
use a common update grid; the proposed cadence is every 1,875 updates. This gives 10 low-presentation checks
and 50 high-presentation checks in both unique-data arms.

The prospective data contract uses one pinned H1000 window manifest, one H200 manifest that is an exact subset
of it, and the same two manifests for initialization seeds 0 and 1. `subset_seed` must no longer be coupled to
`init_seed`.

The exact 2x2 arithmetic is feasible, but D0 recommends fresh training for all eight factorial runs if D1 is
approved. The current checkpoints do not jointly satisfy the shared nested-subset and common validation-cadence
contract. They remain valid historical Route-B anchors.

## Reuse boundary

An existing checkpoint is reusable only if its exact data identities, initialization, update schedule, LR
schedule, validation cadence, and checkpoint-selection opportunity set match the prospective factorial cell.
Existing checkpoints remain authoritative historical Route-B results if they do not satisfy that contract; they
are not silently promoted into D1.

## Outputs

The result package is written to `results/s2p_route_b_phase_d0_schedule_audit/`:

```text
phase_d0_run_schedule.csv
phase_d0_presentations_by_budget.csv
phase_d0_checkpoint_selection.csv
phase_d0_subset_identity.csv
phase_d0_subset_overlap.csv
phase_d0_d1_cell_mapping.csv
phase_d0_reuse_eligibility.csv
phase_d0_go_nogo.json
phase_d0_input_provenance.json
phase_d0_independent_verification.json
phase_d0_artifact_manifest.csv
```

No D1 training is authorized by these artifacts. They only establish the factual schedule and present a
factorial design for PM review.
