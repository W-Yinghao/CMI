#!/bin/bash
# Self-healing submit driver for the 252-cell Readout Ladder CPU array. Excludes cells already in the queue (squeue
# -r %K) as well as .done cells, so no duplicate submissions (see slurm-driver-exclude-running-cells lesson). Only
# submits/polls SLURM (no compute here). Manuscript FROZEN.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
CAP=28
OUTDIR="results/cmi_trace_readout"
mkdir -p "$OUTDIR/cells"
while :; do
  done_n=$(ls "$OUTDIR"/cells/*.done 2>/dev/null | wc -l)
  echo "[ro-driver] done=$done_n/252 $(date -u +%H:%M:%S)"
  [ "$done_n" -ge 252 ] && { echo "[ro-driver] all 252 cells done"; break; }
  running=" $(squeue -u "$USER" -h -r -n ro-ladder -o '%K' 2>/dev/null | tr '\n' ' ') "
  missing=""
  for i in $(seq 0 251); do
    ls "$OUTDIR"/cells/cell_$(printf '%03d' "$i")_*.done >/dev/null 2>&1 && continue
    case "$running" in *" $i "*) continue;; esac
    missing="$missing $i"
  done
  missing=$(echo "$missing" | tr ' ' '\n' | grep -v '^$')
  mine=$(squeue -u "$USER" -h -r -n ro-ladder 2>/dev/null | wc -l)
  room=$((CAP - mine))
  if [ "$room" -ge 1 ] && [ -n "$missing" ]; then
    chunk=$(echo "$missing" | head -n "$room" | paste -sd, -)
    [ -n "$chunk" ] && { sbatch --array="$chunk" scripts/sbatch_readout_label_efficiency.sh >/tmp/ro_sb.$$ 2>&1 && echo "[ro-driver] submitted [$chunk] ($(cat /tmp/ro_sb.$$))" || echo "[ro-driver] refused: $(cat /tmp/ro_sb.$$)"; }
  fi
  sleep 60
done
