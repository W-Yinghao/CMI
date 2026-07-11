#!/usr/bin/env bash
#SBATCH --job-name=fp-gem-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12/logs/smoke-%j.out
#SBATCH --error=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12/logs/smoke-%j.err

set -euo pipefail

REPO="${FP_GEM_REPO:?FP_GEM_REPO must name the clean launch worktree}"
PYTHON=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "$REPO"

test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf 'job_id=%s\n' "${SLURM_JOB_ID:-}"
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
  --mode smoke \
  --dataset BNCI2014_001 \
  --target-subject 1 \
  --seed 0 \
  --out /home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12/smoke.json
