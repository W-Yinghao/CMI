#!/bin/bash
#SBATCH --job-name=s2p9c1
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2/logs/train-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2/logs/train-%A_%a.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_s2p
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
ROOT=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2
TASKS="$ROOT/budget_training_tasks.csv"

ROW=$("$PY" - "$TASKS" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv, sys
tasks = list(csv.DictReader(open(sys.argv[1])))
print(",".join(str(tasks[int(sys.argv[2])][k]) for k in ["cell", "budget_h", "seed", "n_subjects"]))
PY
)
IFS=, read -r CELL H SEED N_SUBJECTS <<< "$ROW"
OUT="$ROOT/$CELL"
mkdir -p "$OUT" "$ROOT/logs"
echo "cell=$CELL H=$H seed=$SEED n_subjects=$N_SUBJECTS host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" s2p/scripts/run_frontier_cbramod.py \
  --allocation-policy high_coverage --min-exposure-hours 0.25 \
  --n-subjects "$N_SUBJECTS" --subset-seed "$SEED" --init-seed "$SEED" --total-hours "$H" \
  --epochs 50 --batch-size 64 --lr 5e-4 --weight-decay 5e-2 --mask-ratio 0.5 --clip-value 1.0 \
  --num-workers 0 --loader-mode streaming --out-dir "$OUT"
