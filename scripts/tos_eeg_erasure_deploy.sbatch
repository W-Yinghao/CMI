#!/bin/bash
#SBATCH --job-name=erase_dep
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python -u -m tos_cmi.eeg.erasure_target_deploy "$@"
echo "ERASE_DEP_SBATCH_DONE"
