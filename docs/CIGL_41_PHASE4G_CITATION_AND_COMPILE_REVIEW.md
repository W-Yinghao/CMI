# CIGL_41 — Phase 4G Citation Closure + LaTeX Compile Smoke Review

> Phase 4G summary (writing/packaging only; no GPU, no training, no experiments, no ablation, no new
> dataset, no λ-grid, no edge-CMI, no SOTA, no CITA/DualPC/Tri-CMI changes, **no PDF/aux/bbl/log committed**,
> no invented citations).

## Citation closure status: **10/10 RESOLVED**

| ref | resolution |
|---|---|
| BNCI2014_001 | `tangermann2012review` (*Front. Neurosci.* 6:55, DOI 10.3389/fnins.2012.00055) + `brunner2008graz2a` |
| BNCI2015_001 | `faller2012autocalibration` (*IEEE TNSRE* 20(3):313–319, DOI 10.1109/TNSRE.2012.2189584) |
| MOABB | `jayaram2018moabb` (*JNE* 15(6):066011, DOI 10.1088/1741-2552/aadea0) |
| EEGNet | `lawhern2018eegnet` (*JNE* 15(5):056013, DOI 10.1088/1741-2552/aace8c) |
| Schirrmeister | `schirrmeister2017deep` (*HBM* 38(11):5391–5420, 2017, DOI 10.1002/hbm.23730) |
| DGCNN | `song2018dgcnn` (*IEEE TAC* 11(3):532–541, 2020 / early access 2018, DOI 10.1109/TAFFC.2018.2817622) |
| RGNN | `zhong2019rgnn` (*IEEE TAC* 13(3):1290–1301, 2022, DOI 10.1109/TAFFC.2020.2994159) — **Crossref-verified** |
| LGGNet | `ding2021lggnet` (*IEEE TNNLS* 35(7):9773–9786, 2024, DOI 10.1109/TNNLS.2023.3236635) — **Crossref-verified**; 3rd author Tong (not Zhang) |
| Li 2018 | `li2018conditional` (*Proc. AAAI* 32(1), 2018, DOI 10.1609/aaai.v32i1.11682) |
| CCMI | `mukherjee2020ccmi` (*PMLR* 115:1083–1093, UAI 2020; no DOI, normal for PMLR) |

Items 1–6 and 9–10 reviewer-verified; **RGNN and LGGNet independently Crossref-verified** (their DOIs
resolve to the correct IEEE records; LGGNet author list and venue/year were corrected). All `\todoverify`
markers removed from the `.tex`. **Only remaining TODO:** the Brunner Graz-2a data-description URL (a
`note`, not required for the citation to be valid). **Nothing fabricated.**

## Remaining TODOs

- One minor non-citation field: `brunner2008graz2a` data-description URL.

## Compile smoke status: **SUCCESS**

- `pdflatex → bibtex → pdflatex ×2` (TeX Live 2024; no `latexmk`). All passes exit 0; BibTeX 11 entries, no
  errors; no undefined citations/references; `_build/main.pdf` (~318 KB, **8 pages**).
- Fix during smoke: moved two `% TODO` comments out of BibTeX entry bodies (in-entry `%` is illegal).
- Generated PDF/aux/bbl/log live only under the **gitignored** `_build/`; `.gitignore` extended for
  `paper/cigl_latex/_build/` + LaTeX artifacts; `*.pdf` already global-ignored. See
  `paper/cigl_latex/COMPILE_SMOKE_SUMMARY.md`.

## Confirmations

- No GPU / no training / no experiments / no ablation / no new dataset / no λ-grid; no edge-CMI; no SOTA; no
  CITA/DualPC/Tri-CMI changes; **no generated PDF/aux/bbl/log committed**; no generated result tables committed.
- Validation: py_compile OK; `test_collect_cigl_evidence_tables`, `test_cigl_manuscript_claims`, and the
  updated `test_cigl_latex_package` (no-PDF/no-artifact, no fabricated DOI, bib parses, cited keys defined)
  pass; collect dry-run OK; `git ls-files` shows no tracked LaTeX artifacts.

## Recommendation (pending reviewer)

**A — ready for human LaTeX prose editing.** Citation closure and compile are done; remaining items are
figure assets, a venue template, the minor data-description URL, and an authorized full PDF build. No
experiment is required; the `graph_010` vs `node_010` ablation stays optional and unauthorized.
