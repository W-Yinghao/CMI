# S2P_42 - Phase D1 Pretraining Closure

**Status:** PRETRAINING PROVENANCE PASS / DOWNSTREAM HELD.

This is a technical experiment record, not manuscript text. It does not authorize D1 downstream analysis,
fine-tuning, H4000, CodeBrain, a new dataset, or submission writing.

## Completed factorial training

The eight preregistered high-horizon trajectories completed the fixed `93,750` optimizer updates:

```text
unique-data arms: 200h, 1000h
subset seeds: 0, 1
initialization seeds: 0, 1
trajectories: 8/8
fixed-update immutable snapshots: 16/16
P_low: update 18,750
P_high: update 93,750
```

All trajectories used batch size 64, the common step-indexed cosine schedule, validation every 1,875 updates,
and diagnostic-only pretraining validation. No early stopping, best-validation checkpoint selection, downstream
labels, or target-task selection entered training.

## Cancellation and exact rerun

Two initial array tasks were intentionally cancelled before either produced a fixed-update snapshot, to release
per-user GPU quota for an unrelated workload. Their partial contracts and logs were preserved outside the
canonical trajectory directories. The same two trajectory identities were restarted from update zero with the
same frozen manifests, initialization hashes, stream seeds, schedule, and output contract. Only the completed
exact reruns enter the canonical 8-trajectory set; no resume or partial state reuse occurred.

## Provenance gate

The independent artifact gate verified:

```text
8/8 PASS_TRAINING completion markers
16/16 content hashes and read-only payloads
16/16 fixed update identities
16/16 strict model/optimizer/scheduler reload records
16/16 bitwise feature reload/repeat checks
same corpus subset across initialization seeds
same initial state and stream contract across paired U arms
same learning rate at matched P_low and P_high updates
target-label and checkpoint-selection firewalls clean
```

Observed aggregate training time was `94.5820 GPU-hours`. Per-trajectory wall time ranged from `10.2322` to
`16.1651` hours across A100 and A40 hardware. Hardware timing is operational metadata, not a scientific factor.

## Disposition

```text
Phase D1 pretraining: CLOSED / PASS
D1 downstream analysis: HELD_UNTIL_PANEL2_CLOSES
Foundation-model fine-tuning: NOT AUTHORIZED
New pretraining: NOT AUTHORIZED
```

The fixed D1 snapshots are now eligible inputs only after the FMScope-FSR Panel 2 primary analysis closes and a
separate downstream protocol gate is approved.

