#!/bin/bash
# Firming-up: submit the E2 TSMNet task-loss re-run when the submit quota frees, then wait for it to finish.
# Touches no other job. Re-runs only TSMNet (EEGNet is head-free; unaffected).
set -uo pipefail
cd "$(dirname "$0")/.."
JOB=e2_tsmnet_taskloss
jid=""
for i in $(seq 1 120); do
  ex=$(squeue -u "$USER" -h -n "$JOB" -o "%i" 2>/dev/null | head -1)
  if [ -n "$ex" ]; then jid=$ex; echo "[$i] found existing $JOB jid=$jid"; break; fi
  out=$(sbatch --parsable --job-name="$JOB" --export=ALL,EXP=e2,BB=TSMNet,NPERM=50 scripts/sbatch_theory_spectrum.sh 2>/dev/null)
  if [ -n "$out" ]; then jid=$out; echo "[$i] SUBMITTED jid=$jid"; break; fi
  sleep 150
done
if [ -z "$jid" ]; then echo "watcher gave up: never got a submit slot in ~5h"; exit 1; fi
# wait for completion
for i in $(seq 1 90); do
  st=$(squeue -j "$jid" -h -o "%T" 2>/dev/null)
  if [ -z "$st" ]; then
    echo "job $jid finished after ~$((i*60))s"
    echo "TSMNet cells: $(ls results/rank_threshold/TSMNet_*.json 2>/dev/null | wc -l)/27"
    exit 0
  fi
  sleep 60
done
echo "job $jid still running after ~90min"; exit 1
