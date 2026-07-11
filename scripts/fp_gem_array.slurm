#!/usr/bin/env bash
#SBATCH --job-name=fp-gem-p12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12/logs/%A_%a.out
#SBATCH --error=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12/logs/%A_%a.err

set -euo pipefail

REPO="${FP_GEM_REPO:?FP_GEM_REPO must name the clean launch worktree}"
GROUP="${FP_GEM_GROUP:?FP_GEM_GROUP must be V100 or A100}"
GROUP_COUNT="${FP_GEM_GROUP_COUNT:?FP_GEM_GROUP_COUNT is required}"
STRIDE="${FP_GEM_GROUP_STRIDE:?FP_GEM_GROUP_STRIDE is required}"
PYTHON=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "$REPO"

if [[ "$GROUP" == "V100" ]]; then
  [[ "$GROUP_COUNT" == "161" && "$STRIDE" == "6" ]]
elif [[ "$GROUP" == "A100" ]]; then
  [[ "$GROUP_COUNT" == "28" && "$STRIDE" == "2" ]]
else
  printf 'unsupported FP_GEM_GROUP=%s\n' "$GROUP" >&2
  exit 2
fi

test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf 'job_id=%s\n' "${SLURM_JOB_ID:-}"
printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-}"
printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID:-}"
printf 'hardware_group=%s\n' "$GROUP"
printf 'hardware_group_count=%s\n' "$GROUP_COUNT"
printf 'shard_stride=%s\n' "$STRIDE"
printf 'host=%s\n' "$(hostname)"
printf 'pwd=%s\n' "$PWD"
printf 'launch_commit=%s\n' "$(git rev-parse HEAD)"
printf 'runner_sha256=%s\n' "$(sha256sum h2cmi/run_fp_gem.py | cut -d' ' -f1)"
printf 'config_sha256=%s\n' "$(sha256sum h2cmi/results/fp_gem_main/fp_gem_config.json | cut -d' ' -f1)"
printf 'repo_status_porcelain_begin\n'
git status --porcelain=v1 --untracked-files=all
printf 'repo_status_porcelain_end\n'
nvidia-smi

for ((group_index=SLURM_ARRAY_TASK_ID; group_index<GROUP_COUNT; group_index+=STRIDE)); do
  printf 'unit_begin group=%s group_index=%s\n' "$GROUP" "$group_index"
  "$PYTHON" -m h2cmi.run_fp_gem \
    --mode run \
    --hardware-group "$GROUP" \
    --group-index "$group_index"
  printf 'unit_complete group=%s group_index=%s\n' "$GROUP" "$group_index"
done

printf 'shard_complete group=%s task=%s\n' "$GROUP" "${SLURM_ARRAY_TASK_ID}"
