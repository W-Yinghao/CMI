#!/bin/bash
#SBATCH --job-name=shumi_dn
#SBATCH --partition=A40,A100,H100,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs/slurm-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs/slurm-%j.err
# --time=8h << 1d cap (multi-partition eligible). MODE + CELLS passed via --export.
set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
echo "host=$(hostname) MODE=${MODE} CELLS=${CELLS}"; nvidia-smi --query-gpu=name --format=csv,noheader || true
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
srun "$PY" /home/infres/yinwang/CMI_AAAI_s2p/s2p/scripts/shumi_downstream_audit.py \
    --mode "${MODE}" --cells ${CELLS} --embedding "${EMB:-spatial}" --norm "${NORM:-patch}" \
    --out-dir "${OUT:-/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream}"
