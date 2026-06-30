#!/bin/bash
#SBATCH --job-name=b3p24c
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=csc/results/b3_p24c.slurm.%j.out
# CSC Route B3-P2.4c studentized subject-consistency gate eval (controls 48 + power 24; dev diagnostic; NO freeze/confirmatory/real EEG).
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_csc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_csc
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
echo "[b3p24c] host=$(hostname) job=${SLURM_JOB_ID:-na} cpus=${SLURM_CPUS_PER_TASK:-na}"
python -u -m csc.mininfo.run_b3_p24c \
  --control_clusters 48 --power_clusters 24 --ms 0 20 30 --n_subjects 36 --n_boot 200 \
  --jobs "${SLURM_CPUS_PER_TASK:-24}" --out csc/results/b3_p24c.json
echo "[b3p24c] done rc=$?"
