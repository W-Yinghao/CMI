#!/bin/bash
#SBATCH --job-name=s2p9c0
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2/logs/feas-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2/logs/feas-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_s2p
mkdir -p results/s2p_budget_floor_calibration_v2/logs
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
OUT=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2

srun "$PY" s2p/scripts/budget_floor_v2_feasibility.py --out-dir "$OUT" --min-exposure-hours 0.25

GO=$("$PY" - <<'PY'
import json
p = "results/s2p_budget_floor_calibration_v2/budget_floor_v2_go_nogo.json"
d = json.load(open(p))
print("1" if d.get("GO") and d.get("auto_launch_training_if_pass") else "0")
PY
)
if [ "$GO" != "1" ]; then
  echo "9C-0 feasibility did not pass; not launching training."
  exit 2
fi

NTASKS=$("$PY" - <<'PY'
import csv
rows = list(csv.DictReader(open("results/s2p_budget_floor_calibration_v2/budget_training_tasks.csv")))
print(len(rows))
PY
)
LAST=$((NTASKS - 1))
echo "9C-0 feasibility PASS; launching ${NTASKS} training tasks."
TRAIN_ID=$(sbatch --parsable --array=0-"$LAST" s2p/scripts/budget_floor_v2_train_array.slurm)
echo "$TRAIN_ID" > "$OUT/train_array_jobid.txt"
PATCH_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" --export=ALL,NORM=patch,REF_ONLY=0 s2p/scripts/budget_floor_v2_downstream.slurm)
echo "$PATCH_ID" > "$OUT/downstream_patch_jobid.txt"
WINDOW_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" --export=ALL,NORM=window,REF_ONLY=1 s2p/scripts/budget_floor_v2_downstream.slurm)
echo "$WINDOW_ID" > "$OUT/downstream_window_jobid.txt"
POST_ID=$(sbatch --parsable --dependency=afterok:"$PATCH_ID":"$WINDOW_ID" s2p/scripts/budget_floor_v2_post.slurm)
echo "$POST_ID" > "$OUT/post_jobid.txt"
echo "Submitted train=${TRAIN_ID} downstream_patch=${PATCH_ID} downstream_window=${WINDOW_ID} post=${POST_ID}"
