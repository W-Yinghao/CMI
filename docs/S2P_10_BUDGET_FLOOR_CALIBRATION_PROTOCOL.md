# S2P_10 - Phase 9C v2 High-Budget Floor Calibration Protocol

**Status:** S2P_10 v1 is superseded. This v2 protocol replaces the fixed-N `{200,500,1000}` plan with a
high-coverage budget ladder that reaches the CodeBrain/CBraMod data-scaling regime. Feasibility, training,
downstream audit, and post-processing are SLURM-only.

## PM decision incorporated

D1 remains accepted as a low-budget floor baseline, not as evidence that subject allocation has no effect on transfer.
The 200 h P1 checkpoints learned subject-identifiable EEG structure, but they did not learn frozen-probe SHU-MI
cross-subject MI structure.

Forbidden claims remain:

```text
subject allocation has no effect on transfer
subject diversity does not matter
200h frontier proves subject count is irrelevant
CBraMod cannot learn MI-transferable representation
pretraining subject diversity has no benefit
```

Allowed framing:

```text
200h is a low-budget floor result under this frozen-probe endpoint.
Subject-identifiable structure appears at 200h.
The next question is the budget threshold for MI-transferable structure.
```

## Primary question

> How much from-scratch CBraMod pretraining budget is needed before frozen, source-only SHU-MI transfer exits the
> random-init floor?

This is a **budget-floor calibration**, not a subject-diversity/depth decomposition and not a released-CBraMod
reproduction claim.

## Budget ladder

Candidate ladder:

```text
H = {200, 500, 1000, 2000, 4000}
```

Training cells:

```text
H = {500, 1000, 2000, H_high}
seeds = {0, 1}
```

Baseline:

```text
H=200 is reused from existing P1 artifacts.
Do not retrain 200h for symmetry.
```

`H_high` selection:

```text
4000h if feasible on the canonical TUEG 19-common corpus;
else the largest feasible 19-common endpoint rounded down to the nearest 500h.
```

The feasibility resolver scans:

```text
4000, 3500, 3000, 2500, 2000
```

If 4000h is infeasible but 3000h or 3500h is feasible, continue automatically with that endpoint. If no feasible
`H_high >= 2000h` exists, stop.

## Allocation rule

Use **high-coverage subject allocation**:

```text
For each budget H:
  choose as many train subjects as possible
  subject to min_exposure_per_subject = 0.25h
  and exact no-reuse window budgeting after fixed pretrain-val exclusion.
```

The PM formula

```text
N_target(H) = min(number of eligible subjects with >=0.25h usable data, floor(H / 0.25))
```

is treated as an upper bound. The implementation resolves the largest exact-window feasible `N` after removing the
fixed subject-disjoint pretrain-val pool. Each selected train subject receives `base` or `base+1` 30 s windows, so:

```text
train_total_windows == round(H * 120)
train_win_max - train_win_min <= 1
no oversampling
no window reuse
```

If the eligible-subject count truncates `N`, actual exposure per subject is recorded as `H/N`.

## Corpus and model

Primary:

```text
model = CBraMod from scratch
corpus = TUEG 19-common only
normalization = per-patch z-score
objective = native CBraMod masked-patch reconstruction
checkpoint selection = pretrain-val loss only
target labels used = false
```

Do not use 33-channel full corpus in this primary curve. A 33-channel feasibility note may be recorded later, but it
cannot patch an infeasible 19-common 4000h point.

Do not include CodeBrain training in 9C v2. CodeBrain remains infrastructure/background only.

## 9C-0 feasibility

Submit:

```text
sbatch s2p/scripts/budget_floor_v2_feasibility.slurm
```

The feasibility job writes:

```text
results/s2p_budget_floor_calibration_v2/budget_grid_feasibility.csv
results/s2p_budget_floor_calibration_v2/high_coverage_subject_plan.csv
results/s2p_budget_floor_calibration_v2/hmax_19common_decision.json
results/s2p_budget_floor_calibration_v2/budget_exposure_table.csv
results/s2p_budget_floor_calibration_v2/pretrain_val_pool_plan.csv
results/s2p_budget_floor_calibration_v2/window_budget_check.csv
results/s2p_budget_floor_calibration_v2/compute_budget_estimate.csv
results/s2p_budget_floor_calibration_v2/slurm_feasibility_report.md
results/s2p_budget_floor_calibration_v2/budget_floor_v2_go_nogo.json
```

`budget_floor_v2_go_nogo.json` includes the PM-required fields:

```json
{
  "phase": "9C_v2_high_budget_floor_calibration",
  "primary_model": "CBraMod",
  "primary_corpus": "TUEG_19_common",
  "candidate_budgets_h": [200, 500, 1000, 2000, 4000],
  "reuse_200h_baseline": true,
  "min_exposure_per_subject_h": 0.25,
  "h4000_feasible_19common": null,
  "h_high_selected_h": null,
  "h_high_selection_rule": "4000_if_feasible_else_largest_19common_endpoint_rounded_down",
  "training_budgets_h": null,
  "n_subjects_by_budget": null,
  "exposure_by_budget": null,
  "subject_disjoint_pretrain_val_feasible": null,
  "exact_window_budget_feasible": null,
  "compute_budget_acceptable": null,
  "target_labels_used": false,
  "auto_launch_training_if_pass": true
}
```

## Auto-launch authorization

PM pre-authorized:

```text
If 9C-0 feasibility passes, automatically launch 9C-1 training.
Do not return for another PM go unless a stop rule triggers.
```

The feasibility SLURM wrapper therefore submits:

```text
9C-1 train array:         s2p/scripts/budget_floor_v2_train_array.slurm
patch downstream audit:   s2p/scripts/budget_floor_v2_downstream.slurm
window reference audit:   s2p/scripts/budget_floor_v2_downstream.slurm
post aggregation:         s2p/scripts/budget_floor_v2_post.slurm
```

The downstream and post jobs are submitted with `afterok` dependencies on the training array.

## 9C-1 training

Each training cell runs:

```text
s2p/scripts/run_frontier_cbramod.py
  --allocation-policy high_coverage
  --min-exposure-hours 0.25
  --loader-mode streaming
  --total-hours H
  --n-subjects N_resolved_by_feasibility
  --subset-seed seed
  --init-seed seed
```

The `streaming` loader is required because the old P1 materialized loader would allocate hundreds of GB at 2000-4000h.
It keeps the native CBraMod model, objective, mask, optimizer, scheduler, loss, and pretrain-val checkpoint rule, while
streaming TUEG rows instead of building one giant in-memory tensor.

Outputs:

```text
results/s2p_budget_floor_calibration_v2/H{H}_s{seed}/best.pth
results/s2p_budget_floor_calibration_v2/H{H}_s{seed}/last.pth
results/s2p_budget_floor_calibration_v2/H{H}_s{seed}/train_log.jsonl
results/s2p_budget_floor_calibration_v2/H{H}_s{seed}/run_summary.json
```

## Downstream audit

After training completes, run the same authoritative SHU-MI frozen audit path used for the final D1 report:

```text
s2p/scripts/shumi_downstream_audit.py
```

Contract:

```text
SHU-MI
19-common channel mapping
native 4s / 4-patch windows
frozen encoder
source-only PCA/head/subspace
target labels final scoring only
patch norm primary
released reference also reported under window norm
```

Required post outputs:

```text
results/s2p_budget_floor_calibration_v2/budget_pretrain_run_manifest.csv
results/s2p_budget_floor_calibration_v2/budget_checkpoint_manifest.csv
results/s2p_budget_floor_calibration_v2/budget_pretrain_logs.csv
results/s2p_budget_floor_calibration_v2/budget_downstream_task_performance.csv
results/s2p_budget_floor_calibration_v2/budget_pairwise_subject_separability.csv
results/s2p_budget_floor_calibration_v2/budget_l4_task_alignment.csv
results/s2p_budget_floor_calibration_v2/budget_l5_replay.csv
results/s2p_budget_floor_calibration_v2/budget_l6_target_consequence.csv
results/s2p_budget_floor_calibration_v2/budget_random_released_references.csv
results/s2p_budget_floor_calibration_v2/budget_floor_summary.json
results/s2p_budget_floor_calibration_v2/budget_target_label_firewall.json
```

## Success criteria

Criterion A:

```text
target bAcc >= random_init + 0.02
```

Criterion B:

```text
target bAcc >= 0.55
```

Criterion C:

```text
source-val task gate passes
```

Interpretation:

```text
If A and C pass:
  from-scratch representation exits random floor.

If A+B+C pass:
  representation reaches weak but usable transfer regime.

If none pass even at H_high:
  current CBraMod/TUEG/SHU-MI frozen-probe path needs released-scale or fine-tuning;
  stop allocation studies.
```

Always report L1 subject separability. If L1 is high before transfer improves, the core science statement is:

```text
subject-identifiable structure emerges earlier than MI-transferable structure.
```

## Stop rules

```text
1. 4000h infeasible and no H_high >= 2000h feasible.
2. exact window budget fails.
3. subject-disjoint pretrain-val pool fails.
4. compute budget exceeds planned cluster limits.
5. target labels appear in any selection.
6. training run hits NaN/Inf unrecoverably.
7. downstream audit pipeline no longer reproduces the released reference sanity.
8. H_high requires 33-channel corpus to be included in the primary curve.
```

If a stop rule triggers, report the artifact and do not continue to the next stage.
