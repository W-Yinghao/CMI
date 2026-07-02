#!/usr/bin/env bash
# Neutral build -> main.pdf. (Venue build wires in via venue_main.tex once venue is fixed; see README.)
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main || true          # bib may be a stub during skeleton phase
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "COMPILE_DONE -> main.pdf"
