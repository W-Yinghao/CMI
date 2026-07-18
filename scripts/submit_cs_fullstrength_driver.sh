#!/bin/bash
# Self-healing submit driver for the Cross-Session 63-bundle 5-arm GPU fleet. Bundle keys precomputed in bash (order
# MUST match tos_cmi.train.run_mcc_arms.enumerate_bundles: BNCI2014 sub1-9 x seed0-2 = idx 0-26, then BNCI2015
# sub1-12 x seed0-2 = idx 27-62). Resubmits MISSING bundles under the QOS cap; exits at 63 .done. sbatch skips .done.
# Only submits/polls SLURM (no compute here). Manuscript FROZEN.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
CAP=28
OUTDIR="results/cmi_trace_cs_fullstrength"
mkdir -p "$OUTDIR"
keys=()
for s in $(seq 1 9);  do for sd in 0 1 2; do keys+=("BNCI2014_001_sub${s}_seed${sd}"); done; done
for s in $(seq 1 12); do for sd in 0 1 2; do keys+=("BNCI2015_001_sub${s}_seed${sd}"); done; done
while :; do
  done_n=$(ls "$OUTDIR"/*_sub*_seed*.done 2>/dev/null | wc -l)
  echo "[csfs-driver] done=$done_n/63 $(date -u +%H:%M:%S)"
  [ "$done_n" -ge 63 ] && { echo "[csfs-driver] all 63 bundles done"; break; }
  missing=""
  for i in $(seq 0 62); do [ -f "$OUTDIR/${keys[$i]}.done" ] || missing="$missing $i"; done
  missing=$(echo "$missing" | tr ' ' '\n' | grep -v '^$')
  mine=$(squeue -u "$USER" -h 2>/dev/null | grep -c cs-fs || true)
  room=$((CAP - mine))
  if [ "$room" -ge 1 ] && [ -n "$missing" ]; then
    chunk=$(echo "$missing" | head -n "$room" | paste -sd, -)
    if [ -n "$chunk" ]; then
      if sbatch --array="$chunk" scripts/sbatch_cs_fullstrength.sh >/tmp/csfs_sb.$$ 2>&1; then
        echo "[csfs-driver] submitted missing [$chunk] ($(cat /tmp/csfs_sb.$$))"
      else
        echo "[csfs-driver] submit refused: $(cat /tmp/csfs_sb.$$)"
      fi
    fi
  fi
  sleep 90
done
