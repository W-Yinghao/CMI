#!/bin/bash
#SBATCH --job-name=fsr_phase4g
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=results/fsr_phase4g_second_moment/slurm_phase4g_%j.out
# NO --time (policy). Single-threaded torch (tiny 32-dim latents). CPU-only, frozen dumps.
set -e
cd /home/infres/yinwang/CMI_AAAI_rq4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "[phase4g] host=$(hostname) start=$(date -Is)"
$PY scripts/run_phase4g_second_moment.py --seeds 0 20260721 20260722 20260723 20260724 20260725 20260726 20260727 20260728 --injtypes varmod covtoken
$PY scripts/aggregate_phase4g_verdict.py
echo "[phase4g] done=$(date -Is)"
