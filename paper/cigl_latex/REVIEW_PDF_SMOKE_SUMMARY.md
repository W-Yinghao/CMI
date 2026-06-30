# CIGL Review-PDF Compile Smoke (Phase 4H / v0.6)

> Local review build only. All generated output goes to the **gitignored** `_build/` and is **not committed**.

## Environment & commands

- `pdflatex` (TeX Live 2024) + `bibtex`; no `latexmk`. Run from `paper/cigl_latex/`.

```
rm -rf _build && mkdir -p _build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
BIBINPUTS="../cigl:.:" bibtex _build/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
```

## Result: **SUCCESS**

- All four passes exit 0; BibTeX 11 entries, no errors.
- **No LaTeX errors; no undefined references or citations; no missing figure files.**
- Now renders the full paper: all **5 tables** (T1–T5) and the **4 figures** — F2/F3 embedded as PNGs from
  real data, F1/F4 as labeled placeholder floats.
- Output: `_build/main.pdf`, ~547 KB, **12 pages** (up from 8 in Phase 4G, which had no tables/figures
  included).

## Warnings

- Routine only (float placement / `hyperref` rerun-to-get-references on intermediate passes, resolved by the
  final pass). No undefined refs in the final log.

## Generated-file hygiene

- Generated PDF: `paper/cigl_latex/_build/main.pdf` — under the **gitignored** `_build/`; **not committed**.
- `_build/` and `*.aux/.bbl/.blg/.log/.out/.fls/.fdb_latexmk/.synctex.gz` are git-ignored; `*.pdf` is
  globally ignored. `git status` shows no build artifacts; committed figure assets are only `.svg`/`.png`
  source files.
- **Confirmation: no PDF / aux / bbl / log committed.**
