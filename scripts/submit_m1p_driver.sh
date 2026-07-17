#!/bin/bash
# Submit the M1-P 63-cell array in chunks that respect the QOS submit cap. Advances a cursor; each chunk fills the
# available room under CAP. Fail-resumable: cells whose .done exists are skipped by the sbatch itself, so
# re-touching is harmless. Exits when all 63 cells have .done markers.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
CAP=28                     # keep my queued+running jobs (incl the login shell) under the ~30 QOS cap
OUTDIR="results/cmi_trace_mechanism_subspace/m1p"
next=0
while :; do
  done_n=$(ls "$OUTDIR"/cell_*.done 2>/dev/null | wc -l)
  echo "[m1p-driver] done=$done_n/63 next=$next $(date -u +%H:%M:%S)"
  [ "$done_n" -ge 63 ] && { echo "[m1p-driver] all 63 cells done"; break; }
  if [ "$next" -le 62 ]; then
    mine=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
    room=$((CAP - mine))
    if [ "$room" -ge 1 ]; then
      end=$((next + room - 1)); [ "$end" -ge 62 ] && end=62
      if sbatch --array=${next}-${end} scripts/sbatch_mechanism_m1p.sh >/tmp/m1p_sb.$$ 2>&1; then
        echo "[m1p-driver] submitted array ${next}-${end} ($(cat /tmp/m1p_sb.$$))"; next=$((end + 1))
      else
        echo "[m1p-driver] submit ${next}-${end} refused: $(cat /tmp/m1p_sb.$$)"   # cap moved; retry next loop
      fi
    fi
  fi
  sleep 90
done
