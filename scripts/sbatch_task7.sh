#!/bin/bash
#SBATCH --job-name=task7_au
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_theoryspectrum
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_theoryspectrum/results/task7_amount_use/slurm_%j.out
# NO -t / walltime (per compute discipline). Resumable: --fleet skips dumps whose JSON exists.
set -e
mkdir -p /home/infres/yinwang/CMI_AAAI_theoryspectrum/results/task7_amount_use
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate icml
python scripts/task7_amount_use.py --fleet --n_folds 12 --k_spec 16 --n_perm 100 --n_random 50 --n_jobs 8
echo "TASK7_SBATCH_WRAPPER_DONE"
