#!/usr/bin/env bash
# AAAI-oriented build -> aaai_main.pdf, and report the page count.
# PHASE 1: article two-column approximation (see aaai_main.tex header). Swap in the
# official aaai27.sty/aaai27.bst at submission. Overfull boxes are expected during
# the quality-first master-draft phase and do NOT fail this script.
set -uo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode aaai_main.tex >/tmp/aaai_p1.log 2>&1
bibtex aaai_main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode aaai_main.tex >/tmp/aaai_p2.log 2>&1
pdflatex -interaction=nonstopmode aaai_main.tex >/tmp/aaai_p3.log 2>&1
pages=$(grep -a "Output written" aaai_main.log | grep -oE '[0-9]+ page' | grep -oE '[0-9]+' | head -1)
undef=$(grep -ciE 'undefined (citation|reference)' aaai_main.log 2>/dev/null || true); undef=${undef:-0}
over=$(grep -c 'Overfull' aaai_main.log 2>/dev/null || true); over=${over:-0}
if [ -f aaai_main.pdf ]; then
  echo "AAAI_COMPILE_DONE -> aaai_main.pdf  pages=${pages:-?}  undefined_refs=${undef}  overfull=${over}"
  echo "  (target: 7 pages main technical content; compression is PHASE 2, not now.)"
else
  echo "AAAI_COMPILE_FAILED (see /tmp/aaai_p3.log)"; exit 1
fi
