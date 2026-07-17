#!/bin/bash
# Submit the MCC 63-bundle GPU array in chunks under the QOS submit cap. Advances a cursor; fail-resumable (the
# sbatch skips any bundle whose .done exists, so re-touching is harmless). Exits when all 63 bundles have .done.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
CAP=28
OUTDIR="results/cmi_trace_mcc"
next=0
while :; do
  done_n=$(ls "$OUTDIR"/*_sub*_seed*.done 2>/dev/null | wc -l)
  echo "[mcc-driver] done=$done_n/63 next=$next $(date -u +%H:%M:%S)"
  [ "$done_n" -ge 63 ] && { echo "[mcc-driver] all 63 bundles done"; break; }
  if [ "$next" -le 62 ]; then
    mine=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
    room=$((CAP - mine))
    if [ "$room" -ge 1 ]; then
      end=$((next + room - 1)); [ "$end" -ge 62 ] && end=62
      if sbatch --array=${next}-${end} scripts/sbatch_mcc_arms.sh >/tmp/mcc_sb.$$ 2>&1; then
        echo "[mcc-driver] submitted array ${next}-${end} ($(cat /tmp/mcc_sb.$$))"; next=$((end + 1))
      else
        echo "[mcc-driver] submit ${next}-${end} refused: $(cat /tmp/mcc_sb.$$)"
      fi
    fi
  fi
  sleep 120
done
