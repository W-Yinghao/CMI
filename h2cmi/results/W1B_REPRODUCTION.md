# W1-B native external reference (BTTA-DG) — reproduction log + result (separate panel; NOT cross-ranked vs W1-A)

## Provenance (frozen third-party snapshot)
upstream: https://github.com/luo-huan-123/BTTA-DG  (ICLR 2026, "Bayesian Test-Time Adaptation via
  Dirichlet Feature Projection and GMM-Driven Inference")
commit:   5932d026bbd8a7de106d31a6d264f4f4924537e4 (master HEAD at clone)
license:  MIT
archive:  BTTA-DG-5932d026.tar  sha256=4de1eb9133bbb731041eb12c743243304e8b7dc639742ca3d0aff1b4148a041c
patch:    W1B_portability.patch (the full diff vs the pinned commit)
env lock: torch 2.8.0+cu128 (repo pin 2.3.1) | moabb 1.2.0 (1.1.1) | mne 1.8.0 | scikit-learn 1.5.2
          (1.5.0) | scipy 1.13.1 | numpy 1.26.4   (env: conda 'icml')
data:     offline datalake /projects/EEG-foundation-model/datalake/raw (no network); moabb 1.2.0 keeps
          the old class-name aliases so download_data.py runs unchanged.

## Protocol (official, unchanged)
LOSO cross-subject (read_mi_combine_tar): source = all non-target subjects, test = held-out subject;
EA covariance alignment; SincAdaptNet ensemble (10 kfolds); official preprocessing (MotorImagery,
session-T, left/right 2-class); official BTTA-DG hyperparameters per dataset (num_components/conf/
entropy). Datasets BNCI2014001 (9), BNCI2014002 (14), BNCI2015001 (12). Unit = target subject.
Reported = Δacc = BTTA-DG − native source-only ENSEMBLE on the SAME pass (so backbone vs adaptation
are separable; absolute accuracy alone is not the comparison).

## Changes made (ALL to make the official code run / measure Δ; none alter the method or its hyperparameters)
1 device      removed hardcoded CUDA_VISIBLE_DEVICES="1" (SLURM allocates the GPU).
2 sharding    env W1B_DATASET / W1B_SUBJ_LO / W1B_SUBJ_HI (parallelism only; LOSO identical).
3 baseline    accumulate the pre-BTTA-DG ensemble argmax on the same pass -> native source-only acc.
4 args.method pretrain's args object lacked .method which their own read_mi_combine_tar reads (their
              BTTA_DG.py already passes method='') -> dataloader-compat.
5 dense-CE    pretrain replicated labels to a FIXED length L=fs*time_length while the augmentation crops
              the signal to a VARIABLE length -> CrossEntropy batch_size mismatch at ep with len<time_
              length. Label length now matches the actual dense output length (intended per-timestep CE
              unchanged).
6 time_length pretrain's default time_length=4 saved models under sub*/4s/ for ALL datasets, but
              BTTA_DG.load_models reads the 512Hz datasets from 5s/ -> FileNotFoundError. Per-dataset
              time_length (4/5/5 = their get_dataset_params) so save path matches load path.
7 diagnostic  log n_gmm_active / n_flip (no behaviour change).
8 intended    env-gated W1B_INTENDED variant ONLY: restore the SINGLE orphaned add_sample() call +
              index the per-trial prob_map 1-D. Default (unset) path is byte-identical to as-published.

## RESULT (both panels; n=35 LOSO subjects)
As-published BTTA-DG  ==  its own source-only ensemble, EXACTLY, on all 35 subjects (Δ +0.0000, harm 0):
    BNCI2014001 9  src/btta 0.8032   BNCI2014002 14  0.7857   BNCI2015001 12  0.7646
ROOT CAUSE (verified): OnlineClustererGMM.add_sample() -- the ONLY method that appends to the GMM
  buffers -- is never called anywhere in the repo; update() only READS the buffer length. So the
  buffers stay empty, the GMM never fits, predict_class() always returns the raw prob_map, and the
  Dirichlet/GMM calibration is an exact NO-OP. Diagnostic: GMM active on 0/N trials for all 35 subjects.
Intended-behavior (orphaned add_sample restored, env-gated): GMM activates on only ~119/5096 trials
  (2.3%) and flips exactly 1 prediction across all 35 subjects; overall Δ +0.0000 (BNCI2014002: 1
  improved / 1 harmed; BNCI2014001 & BNCI2015001 exactly 0). The 32-confident-samples-per-class buffer
  threshold rarely fills in the small online test sets (100-200 trials), so the calibration barely
  engages and almost never changes the argmax.

## Conclusion (W1-B)
In this faithful LOSO reproduction the public BTTA-DG provides NO measurable improvement over its own
source-only SincAdaptNet ensemble (Δ≈0), whether as-published (inert -- orphaned add_sample) or with the
buffer-fill restored (barely-active). The paper's reported gains are not reproduced by the public
snapshot. W1-B is an EXTERNAL end-to-end reference on a DIFFERENT backbone/representation/protocol than
W1-A and is NOT cross-ranked against the same-backbone W1-A panel.
