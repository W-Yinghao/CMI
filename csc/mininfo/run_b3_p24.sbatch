#!/bin/bash
#SBATCH --job-name=b3p24
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=csc/results/b3_p24_dev_map.slurm.%j.out
# CSC Route B3-P2.4 calibrated EXPANDED dev map (pc_centered_calibrated; dev-only; NOT freeze/confirmatory; NO real EEG).
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_csc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_csc
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
echo "[b3p24] host=$(hostname) job=${SLURM_JOB_ID:-na} cpus=${SLURM_CPUS_PER_TASK:-na}"
python -u -m csc.mininfo.run_b3_p24 \
  --clusters 24 --ms 0 10 20 30 --n_subjects 36 --n_boot 200 \
  --jobs "${SLURM_CPUS_PER_TASK:-24}" --out csc/results/b3_p24_dev_map.json
echo "[b3p24] done rc=$?"
