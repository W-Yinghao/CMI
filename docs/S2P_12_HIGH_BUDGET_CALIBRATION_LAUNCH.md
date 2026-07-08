# S2P_12 - High-Budget Calibration Launch Plan

**Phase:** S2P 9C v2 high-budget floor calibration.

**Branch:** `project/s2p-subject-scaling`.

**Result root:**

```text
results/s2p_budget_floor_calibration_v2/
```

## Launch command

The only manual launch command for 9C-0 is:

```text
sbatch s2p/scripts/budget_floor_v2_feasibility.slurm
```

The feasibility job is CPU-only and writes the go/no-go artifacts. If and only if
`budget_floor_v2_go_nogo.json` has `GO=true` and `auto_launch_training_if_pass=true`, the SLURM wrapper submits the
training and downstream dependency chain automatically.

## Automatic dependency chain

```text
9C-0 feasibility
  -> 9C-1 training array: H={500,1000,2000,H_high}, seeds={0,1}
  -> downstream patch audit: all trained H cells + random + released
  -> downstream window reference audit: random + released
  -> post aggregation: budget_* CSV/JSON artifacts
```

The downstream and post jobs use `afterok` dependencies. If any training array task fails, downstream does not run.

## Training implementation

Training uses:

```text
s2p/scripts/run_frontier_cbramod.py
```

with:

```text
--allocation-policy high_coverage
--min-exposure-hours 0.25
--loader-mode streaming
```

The streaming loader is a memory-safety adaptation for 2000-4000h. It preserves the CBraMod architecture, mask,
masked reconstruction objective, optimizer, scheduler, checkpoint selection, and target-label firewall. The old
materialized loader remains the default for the completed P1 path.

## Red-team gates

Before training can auto-launch, the feasibility job must pass:

```text
H_high >= 2000h
exact window budget
train_win_max - train_win_min <= 1
subject-disjoint pretrain-val pool
compute estimate under planned A40 96h cap
target_labels_used == false
19-common primary corpus only
```

After training, the downstream audit must reproduce the random/released sanity references. If released sanity fails,
the post summary is not interpreted as science and the stop rule is triggered.

## No-go behavior

If feasibility fails, the wrapper exits before submitting training. The artifacts to inspect are:

```text
results/s2p_budget_floor_calibration_v2/budget_floor_v2_go_nogo.json
results/s2p_budget_floor_calibration_v2/slurm_feasibility_report.md
results/s2p_budget_floor_calibration_v2/hmax_19common_decision.json
```

No manual rescue should use 33-channel data, oversampled windows, target labels, or CodeBrain training inside this
phase.
