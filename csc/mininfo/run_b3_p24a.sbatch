#!/bin/bash
#SBATCH --job-name=b3p24a
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=csc/results/b3_p24a_controls.slurm.%j.out
# CSC Route B3-P2.4a control-resolution (48 clusters; CONTROLS only; pc_centered_calibrated; dev diagnostic; NO freeze/confirmatory/real EEG).
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_csc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_csc
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
echo "[b3p24a] host=$(hostname) job=${SLURM_JOB_ID:-na} cpus=${SLURM_CPUS_PER_TASK:-na}"
python -u -m csc.mininfo.run_b3_p24a \
  --clusters 48 --ms 0 20 30 --n_subjects 36 --n_boot 200 --base_seed 1000 \
  --jobs "${SLURM_CPUS_PER_TASK:-24}" --out csc/results/b3_p24a_controls.json
echo "[b3p24a] done rc=$?"
