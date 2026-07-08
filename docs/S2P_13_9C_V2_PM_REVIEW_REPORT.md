# S2P_13 - 9C v2 PM Review Report

**Status:** ready for PM review. No SLURM feasibility, training, downstream, or post job has been launched for 9C v2.

**Local commit:** `1247806` (`S2P 9C v2 high-budget calibration launch`)

**Remote status:** local branch is ahead of `origin/project/s2p-subject-scaling` by one commit. Push to
`git@github.com:W-Yinghao/CMI.git` was not performed because the approval reviewer required explicit confirmation for
external GitHub export.

## Decision implemented

S2P_10 v1 is superseded. The new Phase 9C v2 plan tests whether from-scratch CBraMod exits the frozen SHU-MI transfer
floor as pretraining budget increases into the literature-relevant scaling regime.

Primary question:

```text
How much pretraining budget is needed before from-scratch CBraMod exits the frozen-transfer floor?
```

This is not a subject-diversity/depth decomposition and not a released-CBraMod reproduction claim.

## Design now frozen in the launch package

Budget ladder:

```text
200h reused from existing P1 as low-budget floor anchor
500h
1000h
2000h
H_high = 4000h if feasible, else largest feasible canonical 19-common endpoint rounded down
```

Training cells if feasibility passes:

```text
H = {500, 1000, 2000, H_high}
seeds = {0, 1}
model = CBraMod only
corpus = TUEG 19-common only
allocation = high coverage, min 0.25h/subject
```

No 33-channel primary curve, no CodeBrain training, no fine-tuning, no P2 allocation frontier.

## Files added or changed

Protocol and PM-facing docs:

```text
docs/S2P_10_BUDGET_FLOOR_CALIBRATION_PROTOCOL.md
docs/S2P_11_DOWNSTREAM_RESULTS.md
docs/S2P_12_HIGH_BUDGET_CALIBRATION_LAUNCH.md
docs/S2P_13_9C_V2_PM_REVIEW_REPORT.md
```

Implementation:

```text
s2p/scripts/tueg_subject_loader.py
s2p/scripts/run_frontier_cbramod.py
s2p/scripts/shumi_downstream_audit.py
s2p/scripts/budget_floor_v2_feasibility.py
s2p/scripts/budget_floor_v2_post.py
s2p/scripts/budget_floor_v2_feasibility.slurm
s2p/scripts/budget_floor_v2_train_array.slurm
s2p/scripts/budget_floor_v2_downstream.slurm
s2p/scripts/budget_floor_v2_post.slurm
```

Result root placeholder and ignore rules:

```text
results/s2p_budget_floor_calibration_v2/
.gitignore
```

## Execution chain prepared

Manual launch command, only after PM approval:

```text
sbatch s2p/scripts/budget_floor_v2_feasibility.slurm
```

If feasibility passes, the wrapper automatically submits:

```text
training array
patch downstream audit
window released/random reference audit
post aggregation
```

All downstream/post jobs use `afterok` dependencies. If any training task fails, downstream does not run.

## Red-team review

Passes:

```text
no target labels in feasibility/training/checkpoint selection
feasibility inspects only TUEG 19-common metadata/window counts
no 33-channel fallback in the primary curve
no oversampling or window reuse allowed
exact window budget required: train_total_windows == round(H * 120)
train subjects and pretrain-val subjects are disjoint
H_high must be >= 2000h or the phase stops
streaming loader added so 2000-4000h does not materialize hundreds of GB in RAM
training uses explicit A40 96h wall-time under the partition cap
downstream uses the same authoritative SHU-MI audit path as S2P_11
released/random sanity references are rerun in downstream
large checkpoints/embeddings are ignored by git
```

Risks carried:

```text
runtime estimate is extrapolated from 200h P1 logs, not a dry-run benchmark
streaming loader is a new data-feeding path, though objective/model/mask/optimizer/scheduler are unchanged
feasibility has not yet been run, by design
external GitHub push requires explicit user/PM approval
```

## Verification already performed

Local static checks:

```text
PYTHONPYCACHEPREFIX=/tmp/s2p_pycache python -m py_compile \
  s2p/scripts/tueg_subject_loader.py \
  s2p/scripts/run_frontier_cbramod.py \
  s2p/scripts/shumi_downstream_audit.py \
  s2p/scripts/budget_floor_v2_feasibility.py \
  s2p/scripts/budget_floor_v2_post.py

bash -n \
  s2p/scripts/budget_floor_v2_feasibility.slurm \
  s2p/scripts/budget_floor_v2_train_array.slurm \
  s2p/scripts/budget_floor_v2_downstream.slurm \
  s2p/scripts/budget_floor_v2_post.slurm

git diff --check
git diff --cached --check
```

Cluster read-only check:

```text
CPU partition exists with 4 day cap.
A40 partition exists with 4 day cap.
```

## PM decision requested

Approve or reject the 9C v2 launch package.

If approved, the next actions are:

```text
1. Explicitly approve pushing local commit 1247806 to git@github.com:W-Yinghao/CMI.git.
2. Submit sbatch s2p/scripts/budget_floor_v2_feasibility.slurm.
3. If budget_floor_v2_go_nogo.json reports GO=true, allow the pre-authorized auto-launch chain to proceed.
```

If rejected, no compute has been launched and the branch can be revised from the local commit.
