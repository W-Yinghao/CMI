# CIGL LaTeX Compile Smoke (Phase 4G / v0.5)

> Smoke test only. Generated output goes to the **gitignored** `_build/` and is **not committed**.

## Environment

- `pdflatex` = `/usr/bin/pdflatex` (TeX Live 2024); `bibtex` = `/usr/bin/bibtex`. `latexmk` not installed.
- Run from `paper/cigl_latex/`.

## Commands attempted

```
rm -rf _build && mkdir -p _build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
BIBINPUTS="../cigl:.:" bibtex _build/main          # resolves \bibdata{../cigl/REFERENCES_DRAFT}
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
```

(No `latexmk`; used the explicit pdflatex→bibtex→pdflatex×2 sequence.)

## Result: **SUCCESS**

- `pdflatex` pass 1 = 0, `bibtex` = 0, `pdflatex` pass 2 = 0, `pdflatex` pass 3 = 0.
- BibTeX: **11 entries, no errors/warnings** (after moving two `% TODO` comments outside their entries —
  BibTeX does not allow `%` comments inside an entry body).
- **No undefined citations or references** in the final log.
- Output: `_build/main.pdf`, ~318 KB, **8 pages**.

## Fix applied during smoke

The first bibtex run reported 2 errors (`I was expecting a ',' or a '}'`) caused by inline `% TODO` /
`% PMLR` comments placed *inside* `@misc{brunner2008graz2a}` and `@inproceedings{mukherjee2020ccmi}`. Moved
both comments to the line *before* their entries (the BibTeX-legal place for comments). Re-compile is clean.

## Generated-file hygiene

- Generated PDF path: `paper/cigl_latex/_build/main.pdf` — under the **gitignored** `_build/`; **not committed**.
- `_build/` and `*.aux/.bbl/.blg/.log/.out/.fls/.fdb_latexmk/.synctex.gz` under `paper/cigl_latex/` are in
  `.gitignore`; `*.pdf` is already globally ignored. `git status` shows no build artifacts.
- **Confirmation: no PDF / aux / bbl / log committed.**

## To reproduce later (only when authorized for a real build)

From `paper/cigl_latex/`, run the four commands above; the PDF lands in `_build/`.
