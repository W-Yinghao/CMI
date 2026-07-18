#!/bin/bash
# Self-healing submit driver for the Information-Regime Ladder 63-cell CPU array. Resubmits MISSING cells under the
# QOS cap; exits at 63 .done. sbatch skips .done. Only submits/polls SLURM (no compute here). Manuscript FROZEN.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
CAP=28
OUTDIR="results/cmi_trace_info_ladder"
mkdir -p "$OUTDIR/cells"
while :; do
  done_n=$(ls "$OUTDIR"/cells/*.done 2>/dev/null | wc -l)
  echo "[il-driver] done=$done_n/63 $(date -u +%H:%M:%S)"
  [ "$done_n" -ge 63 ] && { echo "[il-driver] all 63 cells done"; break; }
  missing=""
  for i in $(seq 0 62); do
    ii=$(printf "%03d" "$i")
    ls "$OUTDIR"/cells/cell_${ii}_*.done >/dev/null 2>&1 || missing="$missing $i"
  done
  missing=$(echo "$missing" | tr ' ' '\n' | grep -v '^$')
  mine=$(squeue -u "$USER" -h -n info-ladder 2>/dev/null | wc -l)
  room=$((CAP - mine))
  if [ "$room" -ge 1 ] && [ -n "$missing" ]; then
    chunk=$(echo "$missing" | head -n "$room" | paste -sd, -)
    if [ -n "$chunk" ]; then
      if sbatch --array="$chunk" scripts/sbatch_information_ladder.sh >/tmp/il_sb.$$ 2>&1; then
        echo "[il-driver] submitted missing [$chunk] ($(cat /tmp/il_sb.$$))"
      else
        echo "[il-driver] submit refused: $(cat /tmp/il_sb.$$)"
      fi
    fi
  fi
  sleep 60
done
