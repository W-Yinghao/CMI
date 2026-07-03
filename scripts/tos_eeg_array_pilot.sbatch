#!/bin/bash
#SBATCH --job-name=tos_arr
#SBATCH --partition=V100,V100-32GB,V100-16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=tos_arr_%A_%a.out
# Group/array pilot: one array task per manifest line "DATASET BACKBONE SEED".
# Submit with concurrency + group cap:  sbatch --array=0-<K-1>%4 tos_eeg_array_pilot.sbatch <manifest>
# --target-subjects all + idempotent skip-existing => each task fills only its missing folds.
set -e
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
MANIFEST="$1"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
D=$(echo "$LINE" | awk '{print $1}'); BB=$(echo "$LINE" | awk '{print $2}'); S=$(echo "$LINE" | awk '{print $3}')
echo "array task $SLURM_ARRAY_TASK_ID on $(hostname): $D $BB seed$S (fills missing folds)"
/home/infres/yinwang/anaconda3/bin/conda run -n icml --no-capture-output python -u -m tos_cmi.run_eeg_frozen_pilot \
  --dataset "$D" --backbone "$BB" --seed "$S" --target-subjects all --configs erm:0 --epochs 300 --device cuda
echo "ARRAY_TASK_${SLURM_ARRAY_TASK_ID}_DONE"
