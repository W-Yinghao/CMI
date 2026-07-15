#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode tmlr_main.tex
bibtex tmlr_main
pdflatex -interaction=nonstopmode tmlr_main.tex
pdflatex -interaction=nonstopmode tmlr_main.tex
echo "TMLR_COMPILE_DONE -> tmlr_main.pdf"
