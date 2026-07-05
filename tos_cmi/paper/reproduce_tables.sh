#!/usr/bin/env bash
# Regenerate every paper table (.tex) from committed result artifacts. Run from anywhere.
#   bash tos_cmi/paper/reproduce_tables.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${TOS_PY:-python}"
for m in build_table1_summary build_table2_erasure build_table4_bigN; do
  echo ">>> $m"
  "$PY" -m tos_cmi.paper.scripts.$m
done
# Table 3 (frozen-erasure target deployment): needs the committed aggregate summary JSON
SUMM=tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy/erasure_target_deploy_summary.json
if [ ! -f "$SUMM" ]; then
  echo "ERROR: missing $SUMM" >&2
  echo "  regenerate with: sbatch scripts/tos_eeg_erasure_deploy.sbatch --nrandom 8" >&2
  exit 3
fi
echo ">>> build_table3_erasure_target"
"$PY" -m tos_cmi.paper.scripts.build_table3_erasure_target
echo "REPRODUCE_TABLES_DONE -> tos_cmi/paper/figures/table*.tex"
