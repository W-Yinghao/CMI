#!/bin/bash
#SBATCH --job-name=b3p22r1c
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=csc/results/b3_p22_r1c_dev_map.slurm.%j.out
# CSC Route B3-P2.2 R1c DEVELOPMENT map (dev-only; NOT a freeze/confirmatory; NO real EEG).
# No --time (per project compute policy). CPU-only NumPy/sklearn.
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_csc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_csc
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
echo "[b3p22] host=$(hostname) job=${SLURM_JOB_ID:-na} cpus=${SLURM_CPUS_PER_TASK:-na}"
python -u -m csc.mininfo.run_b3_development \
  --clusters 24 --ms 0 5 10 20 30 --n_subjects 36 --min_confirm_pairs 20 \
  --n_boot 200 --jobs "${SLURM_CPUS_PER_TASK:-24}" \
  --out csc/results/b3_p22_r1c_dev_map.json
echo "[b3p22] done rc=$?"
