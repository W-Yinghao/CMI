#!/usr/bin/env bash
# Regenerate every paper figure from committed result artifacts. Run from anywhere.
#   bash tos_cmi/paper/reproduce_figures.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${TOS_PY:-python}"
for m in plot_fig2_synthetic_cert plot_fig3_lpc_collapse plot_fig4_tsmnet_redundant_leakage \
         plot_fig5_eegnet_contrast plot_fig6_factorial; do
  echo ">>> $m"
  "$PY" -m tos_cmi.paper.scripts.$m
done
echo "REPRODUCE_FIGURES_DONE -> tos_cmi/paper/figures/*.pdf"
