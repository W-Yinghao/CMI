#!/bin/bash
#SBATCH --job-name=s2p9cdn
#SBATCH --partition=A40,A100,H100,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2/logs/downstream-%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2/logs/downstream-%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_s2p
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
ROOT=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2
NORM=${NORM:-patch}
REF_ONLY=${REF_ONLY:-0}
if [ "$REF_ONLY" = "1" ]; then
  OUT="$ROOT/downstream_window_ref"
  CELLS="released random"
else
  OUT="$ROOT/downstream_patch"
  CELLS=$("$PY" - <<'PY'
import csv
rows = list(csv.DictReader(open("results/s2p_budget_floor_calibration_v2/budget_training_tasks.csv")))
print(" ".join([r["cell"] for r in rows] + ["random", "released"]))
PY
)
fi
mkdir -p "$OUT" "$ROOT/logs"
if [ -f results/s2p_p1_downstream/p1_channel_mapping_manifest.json ]; then
  cp results/s2p_p1_downstream/p1_channel_mapping_manifest.json "$OUT/p1_channel_mapping_manifest.json"
fi
echo "host=$(hostname) norm=$NORM ref_only=$REF_ONLY cells=$CELLS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" s2p/scripts/shumi_downstream_audit.py \
  --mode D1 --cells $CELLS --embedding spatial --norm "$NORM" \
  --ckpt-root "$ROOT" --out-dir "$OUT" --device cuda:0
