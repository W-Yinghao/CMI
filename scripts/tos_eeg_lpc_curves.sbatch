#!/bin/bash
#SBATCH --job-name=tos_lpc_curves
#SBATCH --partition=V100,V100-32GB,V100-16GB,A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
set -e
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
/home/infres/yinwang/anaconda3/bin/conda run -n icml --no-capture-output python -u -m tos_cmi.run_lpc_curves "$@"
echo "LPC_CURVES_SBATCH_DONE"
