#!/bin/bash
#SBATCH --job-name=b3p23
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=csc/results/b3_p23_dev_map.slurm.%j.out
# CSC Route B3-P2.3 EXPANDED development map (pc_centered LOCKED; dev-only; NOT freeze/confirmatory; NO real EEG).
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_csc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_csc
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
echo "[b3p23] host=$(hostname) job=${SLURM_JOB_ID:-na} cpus=${SLURM_CPUS_PER_TASK:-na}"
python -u -m csc.mininfo.run_b3_p23 \
  --clusters 24 --ms 0 10 20 30 --n_subjects 36 --n_boot 200 \
  --jobs "${SLURM_CPUS_PER_TASK:-24}" --out csc/results/b3_p23_dev_map.json
echo "[b3p23] done rc=$?"
