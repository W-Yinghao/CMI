#!/bin/bash
#SBATCH --job-name=v2s2
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --array=0-4%5
#SBATCH --output=v2s2_shard_%a_%A.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUT=tos_cmi/results/method_deepen/v2_stage2
mkdir -p $OUT/shard_logs
WORLDS=(A B B C C); BBS=(EEGNet EEGNet TSMNet EEGNet TSMNet)
W=${WORLDS[$SLURM_ARRAY_TASK_ID]}; BB=${BBS[$SLURM_ARRAY_TASK_ID]}
echo "=== SHARD $SLURM_ARRAY_TASK_ID : World $W / $BB ===" | tee $OUT/shard_logs/shard_${W}_${BB}.log
echo "HEAD $(git rev-parse HEAD)" | tee -a $OUT/shard_logs/shard_${W}_${BB}.log
echo "config $(sha256sum tos_cmi/eeg/configs/v2_stage2_scoped.yaml | cut -c1-12)" | tee -a $OUT/shard_logs/shard_${W}_${BB}.log
$PY -u -m tos_cmi.eeg.run_v2_stage2_scoped --only-worlds $W --only-backbones $BB \
    --tag shard_${W}_${BB} --outdir $OUT 2>&1 | tee -a $OUT/shard_logs/shard_${W}_${BB}.log
echo "V2S2_SHARD_${W}_${BB}_DONE"
