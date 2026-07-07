# S2P_02 — Parallel Pretraining Infrastructure Plan (Phase 9A)

**Project S2P — Phase 9A.** Job-matrix infrastructure so multi-model subject-scaling pretraining runs in parallel
from day one (not hand-runs). Pretraining code exists and is lab-tested: **CBraMod** (`~/eeg2025/CBraMod/pretrain_main.py`,
`pretrain_trainer.py`, `hbn_datasets.py` — already run on HBN) and **CodeBrain Stage-2** (`~/CodeBrain/Pretrain/pretrain_EEGSSM.py`
+ fixed released TFDual tokenizer). 9A builds the matrix + loader + SLURM array; **9B-0 smoke runs only after PM review.**

## Model matrix (first stage = two models)
```
model_id      substrate                         pretrain objective            channels    notes
M1 CBraMod    CBraMod encoder                    masked reconstruction         200Hz/1s patch, montage-pinned   PRIMARY; deterministic; lab-pretrained
M2 CodeBrain  EEGSSM Stage-2 (tokenizer FROZEN)  masked token prediction       200Hz, TFDual codes             SECONDARY/control; Stage-1 fixed (avoids temporal-collapse variable)
M3 CodeBrain-full  Stage-1+2                     both                          -                                DEFERRED (only on signal)
```

## Run matrix (minimal formal grid; pilot subset)
```
model      ∈ {CBraMod, CodeBrain-Stage2}
condition  ∈ {fixed-hours, growing-hours}
N_subjects ∈ {32, 128, 512, all_or_max}
seed       ∈ {0,1,2}
stage      ∈ {pretrain, feature_dump, downstream_audit}
```
**Pilot (9B-1, post-approval):** `CBraMod × fixed-hours × {32,128,512} × {0,1}`. Infra written for the full matrix.

## run_id + per-run artifacts (unique key)
`run_id = {model_id}_{corpus_subset_id}_{condition}_{n_subjects}_{hours_budget}_{seed}_{stage}`. Each run saves:
```
config.yaml   pretrain_manifest.csv   subject_subset.csv   recording_subset.csv
checkpoint_epoch*.pt   pretrain_log.csv   pretrain_val_subjects.csv
feature_dump_manifest.csv   hashes.json (corpus config, channel-pipeline, subset, code SHA)
```
**Forbidden:** manual subset edits after results; downstream **target** performance for checkpoint selection;
target test labels in any subset decision.

## Checkpoint selection (firewall)
Select epoch by **pretraining-validation loss on held-out pretraining subjects** ONLY. Also report **last epoch**
as a sensitivity. **Never** select by downstream/target performance (even if loss and downstream disagree, primary
checkpoint is loss-selected). `pretrain_val_subjects.csv` are disjoint from the pretraining-train subjects.

## TUEG subject-subset loader (the new piece to build for 9B-0)
Reads `4704743c/TUEG` `metadata.parquet` + `recordings/*.npy` (T_C), selects a **subject subset** (by `subject`),
applies the **pinned channel pipeline** (default: restrict to 19-common-covered recordings; hashed), pins the
**per-subject hours cap** (fixed vs growing), windows to the model's patch spec (CBraMod 1 s/200; CodeBrain 30 s/patches),
and yields batches **without crossing subjects between train and pretrain-val**. Deterministic (seeded).

## SLURM array + parallelization
`s2p/slurm/` array jobs over the run matrix (one array task per run_id), with:
```
resume from last checkpoint; checkpoint every epoch; pretrain-val loss logging each epoch;
feature-dump job after each checkpoint; downstream-audit job depends on feature-dump; claim-safe aggregation last.
```
GPU partitions per `sinfo` (A40/A100/V100). Concurrency capped; per-run seed pins CUDA determinism
(`use_deterministic_algorithms`, cuDNN deterministic) or discloses tolerance. `slurm_array_plan.md` lists the
array indices → run_ids.

## Cost staging (PM)
- **P0 smoke (9B-0):** CBraMod, N∈{32,128}, H0=50 h, seed 0, few epochs — pipeline works, checkpoint saves, feature
  dump + downstream audit run. No science.
- **P1 pilot (9B-1, post-review):** CBraMod × fixed-hours × {32,128,512} × {0,1}, H0=250 h.
- **P2 (post-review):** + CodeBrain-Stage2, + growing-hours, + seed 2, + N=max.
