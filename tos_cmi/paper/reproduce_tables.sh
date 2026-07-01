#!/usr/bin/env bash
# Regenerate every paper table (.tex) from committed result artifacts. Run from anywhere.
#   bash tos_cmi/paper/reproduce_tables.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${TOS_PY:-python}"
for m in build_table1_summary build_table2_erasure; do
  echo ">>> $m"
  "$PY" -m tos_cmi.paper.scripts.$m
done
# Table 3 (frozen-erasure target deployment) -- only if its builder + results exist
if [ -f tos_cmi/paper/scripts/build_table3_erasure_target.py ]; then
  echo ">>> build_table3_erasure_target"
  "$PY" -m tos_cmi.paper.scripts.build_table3_erasure_target
fi
echo "REPRODUCE_TABLES_DONE -> tos_cmi/paper/figures/table*.tex"
