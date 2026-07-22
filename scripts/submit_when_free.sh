#!/bin/bash
# Auto-submit the consolidated theory-spectrum fleet (EXP=all, 1 submit slot) as soon as the per-user submit
# quota frees. Does NOT touch any other job. Exits on first successful submission (or after ~4h of no headroom).
set -uo pipefail
cd "$(dirname "$0")/.."
JOB=cmitrace-theory-all
BASE=38   # my job count when submission first failed with QOSMaxSubmitJobPerUserLimit
for i in $(seq 1 96); do
  if squeue -u "$USER" -h -n "$JOB" 2>/dev/null | grep -q .; then
    echo "[$i] $JOB already queued/running; watcher exiting."; exit 0
  fi
  n=$(squeue -u "$USER" -h -t PD,R 2>/dev/null | wc -l)
  if [ "$n" -lt "$BASE" ]; then
    out=$(sbatch --job-name="$JOB" --export=ALL,EXP=all,KSPEC=16,NPERM=50 scripts/sbatch_theory_spectrum.sh 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "[$i] SUBMITTED (my job count was $n): $out"; exit 0
    fi
    echo "[$i] attempt failed at count=$n: $out"
  else
    echo "[$i] no headroom (my job count=$n >= $BASE); waiting."
  fi
  sleep 150
done
echo "watcher gave up after ~4h: submit quota stayed saturated by other jobs."; exit 1
