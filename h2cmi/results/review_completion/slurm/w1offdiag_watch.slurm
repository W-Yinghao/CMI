#!/usr/bin/env bash
#SBATCH --job-name=w1offdiag-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.err

set -euo pipefail

cd /home/infres/yinwang/CMI_AAAI_qxu
mkdir -p h2cmi/results/review_completion/slurm/logs h2cmi/results/review_completion

STATUS=h2cmi/results/review_completion/offdiag_watch_status.md
WATCH_JOBS=888742,888743

{
  echo "# Off-Diagonal Geometry Watch Status"
  echo
  echo "- watcher_job: ${SLURM_JOB_ID:-unknown}"
  echo "- watched_jobs: ${WATCH_JOBS}"
  echo "- started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
} > "$STATUS"

while true; do
  {
    echo
    echo "## $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    squeue -j "$WATCH_JOBS" -o '%18i %24j %10T %12M %10l %12P %R' || true
    echo
    find results/h2cmi/review_completion_offdiag -maxdepth 1 -type f -printf '%f %s bytes\n' | sort || true
  } >> "$STATUS"

  if ! squeue -h -j "$WATCH_JOBS" | grep -q .; then
    break
  fi
  sleep 300
done

{
  echo
  echo "## Final Queue Check"
  echo "- completion_rule: job absent from squeue + artifact parse/count/checksum validation passed"
  if squeue -h -j "$WATCH_JOBS" | grep -q .; then
    echo "- final_squeue_absent: false"
    squeue -j "$WATCH_JOBS" -o '%18i %24j %10T %12M %10l %12P %R' || true
  else
    echo "- final_squeue_absent: true"
  fi
  echo
  echo "## Final Artifacts"
  ls -lh results/h2cmi/review_completion_offdiag/ h2cmi/results/review_completion/geometry_capacity_offdiagonal_results.csv 2>/dev/null || true
  echo
  echo "- finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$STATUS"
