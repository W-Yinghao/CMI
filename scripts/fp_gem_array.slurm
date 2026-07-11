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
PYTHON=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "$REPO"

test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf 'job_id=%s\n' "${SLURM_JOB_ID:-}"
printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-}"
printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID:-}"
printf 'hardware_group=%s\n' "$GROUP"
printf 'host=%s\n' "$(hostname)"
printf 'pwd=%s\n' "$PWD"
printf 'launch_commit=%s\n' "$(git rev-parse HEAD)"
printf 'runner_sha256=%s\n' "$(sha256sum h2cmi/run_fp_gem.py | cut -d' ' -f1)"
printf 'config_sha256=%s\n' "$(sha256sum h2cmi/results/fp_gem_main/fp_gem_config.json | cut -d' ' -f1)"
printf 'repo_status_porcelain_begin\n'
git status --porcelain=v1 --untracked-files=all
printf 'repo_status_porcelain_end\n'
nvidia-smi

exec "$PYTHON" -m h2cmi.run_fp_gem \
  --mode run \
  --hardware-group "$GROUP" \
  --group-index "${SLURM_ARRAY_TASK_ID}"
