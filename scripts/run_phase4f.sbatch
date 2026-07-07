#!/bin/bash
#SBATCH --job-name=fsr_phase4f
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=results/fsr_phase4f_corrected_repair/slurm_phase4f_%j.out
# NO --time (policy). Single-threaded torch (tiny 32-dim latents). CPU-only, frozen dumps.
set -e
cd /home/infres/yinwang/CMI_AAAI_rq4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "[phase4f] host=$(hostname) start=$(date -Is)"
$PY scripts/run_phase4f_corrected_repair.py --seeds 0 20260721 20260722 20260723 20260724 20260725 20260726 20260727 20260728
$PY scripts/aggregate_phase4f_verdict.py
echo "[phase4f] done=$(date -Is)"
