#!/bin/bash
#SBATCH --job-name=v2s2merge
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --output=v2s2merge_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python -u -m tos_cmi.eeg.merge_v2_stage2 \
    --outdir tos_cmi/results/method_deepen/v2_stage2 --tag stage2
echo "V2S2MERGE_SBATCH_DONE"
