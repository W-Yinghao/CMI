#!/bin/bash
# Mechanism-Subspace Oracle FULL M1 run (126 cells) — AMENDMENT 03 (shared-null conditional estimand). CPU only
# (frozen features + backfilled DGCNN session; sklearn logistic + numpy eigensolves, no GPU training). PREPARED but
# HELD: full M1 is on HOLD until the PM reviews D1-D5. Submit ONLY after the project owner/PM releases M1. DGCNN
# session sidecars must be backfilled first (scripts/backfill_dgcnn_session.py). Manuscript FROZEN. Only the
# project owner may stop a scientific line.
# NOTE (M1 execution design, PM section 7): the released M1 should run as a FAIL-RESUMABLE SLURM ARRAY over the 126
# fold-seed cells (each task writes its own cell row + random rows + completeness marker + hashes), aggregating
# ONLY after all 126 complete (no threshold edits on partial results). This monolithic form is the smoke/dev
# driver; the array wrapper is finalized when M1 is released.
#SBATCH --job-name=mech-oracle-m1
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/mech/mech-oracle-m1-%j.out
#SBATCH --error=logs/mech/mech-oracle-m1-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/mech
# process-level parallelism, intra-op threads pinned to 1 (per project compute policy)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3/bin/python
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
echo "config_hash=$("$PY" -c "from tos_cmi.eval.mechanism_subspace import config_hash; print(config_hash('configs/cmi_trace_mechanism_subspace_oracle_v4.yaml'))")"
# Backfill DGCNN session sidecars (strict, zero-GPU) so DGCNN runs the SAME session-macro split as EEGNet.
"$PY" scripts/backfill_dgcnn_session.py || { echo "FATAL: DGCNN session backfill failed"; exit 1; }
# FULL M1: 2 datasets x 2 backbones x (9+12) subjects x seeds 0,1,2 x 4 families; SHARED_NULL_HAAR primary control
# 2 blocks x 50 + ambient 2x50 (A03.2). Confirmatory primary = contrast/EEGNet/both datasets, EXACT sign-flip p.
"$PY" scripts/run_mechanism_subspace_oracle.py --seeds 0 1 2 --n_random 50 --blocks 2
"$PY" scripts/aggregate_mechanism_subspace_oracle.py
echo "[mech-oracle-m1] done; verdict -> results/cmi_trace_mechanism_subspace/mechanism_oracle_verdict_full.json"
