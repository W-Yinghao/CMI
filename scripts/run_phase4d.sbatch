#!/bin/bash
#SBATCH --job-name=fsr_phase4d
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=results/fsr_phase4d_repair/slurm_phase4d_%j.out
# NO --time (per project policy). Single-threaded torch (tiny 32-dim latents); CPU-only, frozen dumps.
set -e
cd /home/infres/yinwang/CMI_AAAI_rq4
export PYTHONHASHSEED=0          # reproducible hash-based token/c_target (recorded in manifest)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "[phase4d] host=$(hostname) start=$(date -Is) PYTHONHASHSEED=$PYTHONHASHSEED"
$PY scripts/run_phase4d_repair.py --alphas 1.0 2.0
$PY scripts/aggregate_phase4d_verdict.py
echo "[phase4d] done=$(date -Is)"
