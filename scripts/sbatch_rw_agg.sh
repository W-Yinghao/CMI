#!/bin/bash
# RW-MCC full aggregation (CPU): DG utility + the heavy source-LOSO excess-risk recompute on the A/B arm dumps
# (does RW-MCC reduce the source transfer gap it targets?). env c84c (reads npz dumps + sklearn; no braindecode).
# Submit via SLURM, not the login node. Manuscript FROZEN.
#SBATCH --job-name=rw-agg
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/mcc/rw-agg-%j.out
#SBATCH --error=logs/mcc/rw-agg-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/mcc
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PY=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3/bin/python
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
"$PY" scripts/aggregate_rw_mcc.py --from-dir results/cmi_trace_rw_mcc --expect 63
echo "[rw-agg] done -> results/cmi_trace_rw_mcc/rw_mcc_verdict.json"
