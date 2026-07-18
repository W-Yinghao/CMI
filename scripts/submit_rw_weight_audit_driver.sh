#!/bin/bash
# Self-healing submit driver for the MCC estimator-audit 63-cell array. Each loop computes the MISSING bundle
# indices (0-62 without a .done) and submits up to (CAP - my_jobs) of them as a comma-list array. Resubmits cells
# that failed (e.g. a bad GPU) instead of stalling on a cursor. Exits when all 63 .done. The sbatch skips .done.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
CAP=28
OUTDIR="results/cmi_trace_risk_weighted_mcc/wcells"
while :; do
  done_n=$(ls "$OUTDIR"/cell_*.done 2>/dev/null | wc -l)
  echo "[rwwaudit-driver] done=$done_n/63 $(date -u +%H:%M:%S)"
  [ "$done_n" -ge 63 ] && { echo "[rwwaudit-driver] all 63 cells done"; break; }
  # missing = indices 0..62 whose zero-padded cell_NNN_*.done does not exist (loop-based; no fragile comm sort)
  missing=""
  for i in $(seq 0 62); do
    compgen -G "$OUTDIR/cell_$(printf '%03d' "$i")_*.done" > /dev/null || missing="$missing $i"
  done
  missing=$(echo "$missing" | tr ' ' '\n' | grep -v '^$')
  mine=$(squeue -u "$USER" -h 2>/dev/null | grep -c rw-waudit || true)
  room=$((CAP - mine))
  if [ "$room" -ge 1 ] && [ -n "$missing" ]; then
    chunk=$(echo "$missing" | head -n "$room" | paste -sd, -)
    if [ -n "$chunk" ]; then
      if sbatch --array="$chunk" scripts/sbatch_rw_weight_audit.sh >/tmp/audit_sb.$$ 2>&1; then
        echo "[rwwaudit-driver] submitted missing [$chunk] ($(cat /tmp/audit_sb.$$))"
      else
        echo "[rwwaudit-driver] submit refused: $(cat /tmp/audit_sb.$$)"
      fi
    fi
  fi
  sleep 90
done
