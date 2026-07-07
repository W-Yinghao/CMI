#!/bin/bash
#SBATCH --job-name=fact_ms
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
# pin per-process BLAS/OpenMP to 1 thread so joblib file-level parallelism does not oversubscribe cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python -u -m tos_cmi.eeg.factorial_multiseed_analysis
echo "FACT_MS_SBATCH_DONE"
