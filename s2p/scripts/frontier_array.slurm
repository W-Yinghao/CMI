#!/bin/bash
#SBATCH --job-name=fr_p1
#SBATCH --partition=A40,A100,H100,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --array=0-14
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_cbramod/logs/slurm-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_cbramod/logs/slurm-%A_%a.err
# NO --time (project discipline). Multi-partition: SLURM schedules each of the 15 cells on whichever GPU is free
# (A40 48GB / A100 40-80GB / H100 80GB / V100 16-32GB). CBraMod is 4.9M params, ~4-8GB -> fits V100-16GB.
set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1

# 15 cells = 5 N x 3 seeds (subset_seed == init_seed per manifest). i = SLURM_ARRAY_TASK_ID.
NS=(128 128 128 256 256 256 512 512 512 1024 1024 1024 2048 2048 2048)
SS=(0   1   2   0   1   2   0   1   2   0    1    2    0    1    2)
i=${SLURM_ARRAY_TASK_ID}
N=${NS[$i]}; S=${SS[$i]}
OUT=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_cbramod/N${N}_s${S}
mkdir -p "$OUT"
echo "task=$i N=$N seed=$S host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
srun "$PY" /home/infres/yinwang/CMI_AAAI_s2p/s2p/scripts/run_frontier_cbramod.py \
    --n-subjects "$N" --subset-seed "$S" --init-seed "$S" --total-hours 200 \
    --epochs 50 --batch-size 64 --lr 5e-4 --weight-decay 5e-2 --mask-ratio 0.5 --clip-value 1.0 \
    --num-workers 6 --out-dir "$OUT"
