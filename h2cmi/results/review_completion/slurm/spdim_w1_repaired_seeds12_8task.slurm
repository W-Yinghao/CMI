#!/usr/bin/env bash
#SBATCH --job-name=spdim-p9-s12
#SBATCH --partition=H100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%4
#SBATCH --output=h2cmi/results/review_completion/slurm/logs/%x-%A_%a.out
#SBATCH --error=h2cmi/results/review_completion/slurm/logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
EXT=/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39
MANIFEST=h2cmi/results/review_completion/w1_repaired_split_manifest.csv
MANIFEST_HASH=231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e
LAUNCH_COMMIT=$(git rev-parse HEAD)
OUT_ROOT=/home/infres/yinwang/.cache/h2cmi_training_caches/p9_repaired_seeds12_${LAUNCH_COMMIT}

SPECS=(
  "BNCI2014_001=1-9;Cho2017=1-20"
  "Cho2017=21-49"
  "Cho2017=50-52;Lee2019_MI=1-26"
  "Lee2019_MI=27-54"
)
EXPECTED_ROWS=(116 116 116 112)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SEED=$((TASK_ID / 4 + 1))
SHARD_INDEX=$((TASK_ID % 4))
TARGET_SPEC=${SPECS[$SHARD_INDEX]}
EXPECTED=${EXPECTED_ROWS[$SHARD_INDEX]}
SHARD=seed${SEED}_shard${SHARD_INDEX}

export PYTHONPATH="$PWD:$EXT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export MPLCONFIGDIR=/tmp/h2cmi-p9-mpl-${SLURM_JOB_ID}-${TASK_ID}
mkdir -p "$MPLCONFIGDIR" "$OUT_ROOT" h2cmi/results/review_completion/slurm/logs

echo "job_id=${SLURM_JOB_ID}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=$TASK_ID"
echo "seed=$SEED"
echo "shard_index=$SHARD_INDEX"
echo "shard_id=$SHARD"
echo "host=$(hostname)"
echo "pwd=$PWD"
echo "launch_commit=$LAUNCH_COMMIT"
echo "manifest_hash=$MANIFEST_HASH"
echo "target_spec=$TARGET_SPEC"
echo "expected_rows=$EXPECTED"
echo "out_root=$OUT_ROOT"
echo "controller_sha256=$(sha256sum h2cmi/run_spdim_w1_repaired_seeds12.py | awk '{print $1}')"
echo "runner_sha256=$(sha256sum h2cmi/run_spdim_w1_repaired_seed0.py | awk '{print $1}')"
echo "config_sha256=$(sha256sum h2cmi/config.py | awk '{print $1}')"
echo "manifest_csv_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')"
echo "external_sha=$(git -C "$EXT" rev-parse HEAD)"
echo "repo_status_porcelain_begin"
git status --porcelain=v1 --untracked-files=all
echo "repo_status_porcelain_end"
nvidia-smi || true

"$PY" -m h2cmi.run_spdim_w1_repaired_seeds12 \
  --mode run \
  --external-spdim-path "$EXT" \
  --datasets BNCI2014_001,Cho2017,Lee2019_MI \
  --manifest "$MANIFEST" \
  --manifest-hash "$MANIFEST_HASH" \
  --target-spec "$TARGET_SPEC" \
  --shard-id "$SHARD" \
  --seed "$SEED" \
  --epochs 20 \
  --adapt-epochs 30 \
  --adapt-lr 0.01 \
  --batch-size 64 \
  --val-fraction 0.2 \
  --temporal-filters 4 \
  --spatial-filters 40 \
  --subspace-dims 20 \
  --dtype float32 \
  --spd-device cpu \
  --device cuda \
  --out "$OUT_ROOT/${SHARD}_results.csv" \
  --audit "$OUT_ROOT/${SHARD}_audit.md" \
  --summary "$OUT_ROOT/${SHARD}_summary.json" \
  --digest "$OUT_ROOT/${SHARD}_result_digest.md" \
  --legacy-compare "$OUT_ROOT/${SHARD}_legacy_compare.md" \
  --failure-trace "$OUT_ROOT/${SHARD}_failure_trace.txt"
