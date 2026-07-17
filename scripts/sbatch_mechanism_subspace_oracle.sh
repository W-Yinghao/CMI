#!/bin/bash
# Mechanism-Subspace Oracle FULL M1 run (126 cells) — CPU only (frozen features; sklearn logistic + numpy
# eigensolves, no GPU training). PREPARED but HELD: full M1 is on HOLD until Stage B/C are reviewed. Submit ONLY
# after the project owner/PM releases M1. Manuscript FROZEN. Only the project owner may stop a scientific line.
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
echo "config_hash=$("$PY" -c "from tos_cmi.eval.mechanism_subspace import config_hash; print(config_hash('configs/cmi_trace_mechanism_subspace_oracle_v3.yaml'))")"
# FULL M1: 2 datasets x 2 backbones x (9+12) subjects x seeds 0,1,2 x 4 families; random ambient 100 (2x50),
# shared-overlap-matched pool 5000 (P0.5). Confirmatory primary = contrast/EEGNet/both datasets.
"$PY" scripts/run_mechanism_subspace_oracle.py --seeds 0 1 2 --n_random 50 --blocks 2 --pool 5000
"$PY" scripts/aggregate_mechanism_subspace_oracle.py
echo "[mech-oracle-m1] done; verdict -> results/cmi_trace_mechanism_subspace/mechanism_oracle_verdict_full.json"
