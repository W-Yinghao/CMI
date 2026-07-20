#!/bin/bash
#SBATCH --job-name=c87_ctrl
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_oaci/c87_control_out/slurm_%j.out
set -e
source ~/.bashrc
conda activate c84c-eeg2025-v3-exact
cd /home/infres/yinwang/CMI_AAAI_oaci
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -m oaci.active_testing.c87.run_control_gate /home/infres/yinwang/CMI_AAAI_oaci/c87_control_out
echo "EXIT_CODE=$?"
