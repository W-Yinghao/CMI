# S2P_10 - Budget-Floor Calibration Protocol (FROZEN; pending PM compute go)

**Status:** protocol-only. No feasibility job, CPU analysis job, GPU training job, or downstream job has been launched
for this phase. After this file is reviewed, every experimental action in this phase, including CPU-only feasibility
and summary generation, must run through SLURM (`sbatch`). Local shell use is limited to file inspection, Git, and
documentation edits.

## PM decision incorporated

D1 is accepted as a low-budget floor baseline, not as evidence that the target-transfer allocation slope is truly flat.
The 200 h P1 checkpoints learned the masked-reconstruction objective enough to encode subject-identifiable EEG
structure, but they did not learn SHU-MI cross-subject MI structure that transfers under a frozen source-only probe.

Allowed framing:

```text
pretraining objective learned: yes
subject-identifiable structure learned: yes
cross-subject MI-transferable structure learned: no, under this frozen-probe endpoint
allocation frontier target-transfer slope: not meaningful because the endpoint sits at floor
```

Forbidden claims:

```text
subject allocation has no effect on transfer
subject diversity does not matter
200h frontier proves subject count is irrelevant
CBraMod cannot learn MI-transferable representation
pretraining subject diversity has no benefit
```

## Primary question

> At what from-scratch CBraMod pretraining budget does frozen, source-only SHU-MI cross-subject MI transfer first rise
> above the random-init floor?

This is a **budget-floor calibration**, not a new subject-diversity or allocation-frontier claim. Its purpose is to
decide whether S2P should ever resume allocation-frontier testing at a larger budget.

## Design

Primary candidate:

```text
N = 1024
H0 in {200h, 500h, 1000h}
seeds = {0, 1}
model = CBraMod from scratch
endpoint = frozen encoder + source-only SHU-MI probe, patch norm primary
```

Rationale: N=1024 is a middle/high-subject point that avoids the N=128 long-recording clinical endpoint and the
N=2048 ultra-shallow endpoint.

Fallback candidate:

```text
N = 512
H0 in {200h, 500h, 1000h}
seeds = {0, 1}
```

Fallback is used only if a SLURM-submitted feasibility manifest shows N=1024 is infeasible for the required H0 ladder.
Do not silently switch N.

The 200 h baseline should reuse existing P1 N1024_s0/s1 (or N512_s0/s1 if fallback is activated) artifacts unless a
feasibility or metadata audit finds a mismatch. Do not retrain the 200 h baseline just for symmetry.

## Feasibility gate

Before any pretraining launch, submit a CPU SLURM feasibility job that materializes only manifests from
`tueg_subject_loader.build_frontier_cell(N, subset_seed, total_hours=H0)`. The job must write:

```text
results/s2p_budget_floor/feasibility_manifest.csv
results/s2p_budget_floor/feasibility_verdict.json
```

Required fields:

```text
N, H0, seed, WT, exposure_h, base_windows, plus1_subjects, need_w,
pool_size_after_val_exclusion, train_total_windows, train_total_hours,
pct_off_budget, train_win_min, train_win_max, train_win_maxmin,
train_val_disjoint, verdict
```

PASS criteria:

```text
pool_size_after_val_exclusion >= N
pct_off_budget == 0
train_win_maxmin <= 1
train_val_disjoint == true
```

If N=1024 fails for any required H0, run the same feasibility gate for N=512 and return the manifest before training.

## Pretraining stage

Use the existing native CBraMod runner path, with `--total-hours` set by H0:

```text
s2p/scripts/run_frontier_cbramod.py
```

Output root:

```text
results/s2p_budget_floor/N{N}_H{H0}_s{seed}/
```

Training contract:

```text
native CBraMod architecture/objective/mask
per-patch z-score
HBN normalizer neutralized
no /100 scale change
checkpoint selected by pretrain-val loss only
target labels never used
```

Runtime contract:

```text
all training via sbatch
set --time explicitly under the smallest eligible partition cap
if 500h/1000h cannot fit a single SLURM wall-time cap, do not use no-time jobs;
instead add a checkpoint-resume/chunking implementation and return it for PM review before launch
```

## Probe gate and fleet

After the first new-budget checkpoint is available, run one downstream probe through SLURM before launching the full
calibration fleet. The probe should include:

```text
one new-budget cell, preferably N1024_H500_s0
fixed random-init floor
fixed released-CBraMod reference
```

Gate checks:

```text
channel map exact
native 4-patch downstream forward
embedding deterministic
source-only PCA/head/rank
target-label firewall clean
metrics and variance null complete
released reference remains above random
```

If the probe gate fails, stop and report. If it passes, run the remaining calibration downstream cells by SLURM.

## Primary criterion

A budget is considered to have cleared the frozen-transfer floor only if:

```text
mean target bAcc at that H0 >= random-init target bAcc + 0.02
source-val gate passes
released-reference remains above random under the same audit
```

Random-init and released-reference controls are fixed references. If any audit code changes affect their values, rerun
them through SLURM and report the delta.

## Secondary endpoints

Report, but do not use as floor-crossing criteria:

```text
L1 pairwise subject separability
L4 task alignment
L5 subject-subspace removal vs variance-matched null, only if task gate passes
pretrain-val loss and convergence
population/redundancy manifest
```

## Decision rules

If 500 h or 1000 h clears the floor:

```text
budget threshold is bracketed;
return to PM before any allocation frontier at that budget
```

If neither 500 h nor 1000 h clears the floor:

```text
from-scratch CBraMod under this frozen-probe protocol likely needs released-scale pretraining;
do not run S2P allocation P2
```

Fine-tuning sanity is not part of this main phase. If approved later, it must be named `fine_tune_sanity`, reported
separately, and framed as initialization quality under source-supervised fine-tuning, not frozen representation quality.

## Outputs

Protocol-gated outputs:

```text
docs/S2P_10_BUDGET_FLOOR_CALIBRATION_PROTOCOL.md
results/s2p_budget_floor/feasibility_manifest.csv
results/s2p_budget_floor/feasibility_verdict.json
results/s2p_budget_floor/pretrain_manifest.csv
results/s2p_budget_floor/downstream_raw.csv
results/s2p_budget_floor/budget_floor_summary.json
docs/S2P_12_BUDGET_FLOOR_CALIBRATION_RESULTS.md
```

The results doc must be written and committed before interpretation. After the results doc, return to PM and wait.

## Stop rules

```text
1. Any CPU/GPU experimental job is run outside SLURM.
2. Feasibility manifest is missing or fails.
3. Target labels are used for anything except final scoring.
4. 500h/1000h training is launched with no explicit --time.
5. Runtime does not fit partition caps and no reviewed resume/chunking plan exists.
6. The phase is reframed as a subject-diversity/allocation test.
7. P2 allocation frontier, CodeBrain, new datasets, or fine-tuning are added without explicit PM approval.
```
