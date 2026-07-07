#!/bin/bash
#SBATCH --job-name=fsr_phase4e
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=results/fsr_phase4e_token_centering/slurm_phase4e_%j.out
# NO --time (project policy). Single-threaded torch (tiny 32-dim latents). CPU-only, frozen dumps.
set -e
cd /home/infres/yinwang/CMI_AAAI_rq4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "[phase4e] host=$(hostname) start=$(date -Is)"
$PY scripts/run_phase4e_token_centering.py --seeds 0 20260707 20260708 20260709
$PY scripts/aggregate_phase4e_verdict.py
echo "[phase4e] done=$(date -Is)"
