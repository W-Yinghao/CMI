# paper/cigl_latex/ — CIGL LaTeX-ready package (Phase 4H / v0.6)

**LaTeX-ready draft, NOT a submission and NOT compiled.** This converts the approved Markdown v0.3
manuscript (`../cigl/MANUSCRIPT_DRAFT.md`) into a LaTeX skeleton a human can edit. Writing/packaging only.

## Hard rules
- **No PDF compilation** without explicit reviewer authorization (no `.pdf` is committed).
- **No new experiments / GPU / training / ablation / dataset / λ-grid / edge-CMI / SOTA.**
- **No CITA/DualPC/Tri-CMI changes.**
- **No invented citations.** Bib entries (`../cigl/REFERENCES_DRAFT.bib`) carry only verified-identity
  fields; unverified venue/DOI/pages are `% TODO` comments, never fabricated values.
- **Scientific claims are fixed and bounded** (see `../cigl/CLAIMS_AUDIT.md`): posterior-KL proxy (not
  unbiased CMI), graph/node only (no edge object/edge-CMI), DGCNN static adjacency, source-only,
  `λ_g=λ_n=0.010`, two MI datasets, **partial** reduction at source-task retention, no SOTA, no beyond-MI.

## Layout
- `main.tex` — neutral `article` class (no conference template yet); abstract + `\input` of sections + bib.
- `sections/01..07*.tex` — Intro(+contributions), Related Work, Method, Protocol, Results, Analysis &
  Negative Results, Limitations & Conclusion.
- `tables/table1..5*.tex` — method/protocol, leakage audit, BNCI2014_001, BNCI2015_001, negative results
  (LaTeX tabular drafts; values from `../cigl/RESULTS_TABLES_DRAFT.md` + `STATISTICAL_SUMMARY_DRAFT.md`).
- `figures/FIGURE_ASSET_PLAN.md` — F1–F4 plans (no images generated).

## Citations
`\citep{key}` resolves all 10 references; **all citations are now resolved** (Phase 4G) — see
`../cigl/REFERENCES_DRAFT.bib`, `../cigl/REFERENCES_VERIFIED.md`, `../cigl/CITATION_TODO_QUEUE.md`. RGNN and
LGGNet DOIs/authors were Crossref-verified. The only remaining TODO is a minor `note` field (Brunner Graz-2a
data-description URL). No `\todoverify` markers remain.

## Compile smoke (Phase 4G): PASSED
`main.tex` compiles end-to-end (pdflatex → bibtex → pdflatex ×2): 11 entries, no errors, no undefined
citations, `_build/main.pdf` ~8 pages. See `COMPILE_SMOKE_SUMMARY.md`. The generated PDF and aux/bbl/log are
written to the **gitignored** `_build/` and are **not committed**.

## To build later (only when authorized for a real submission build)
From this directory: `pdflatex -output-directory=_build main && BIBINPUTS="../cigl:.:" bibtex _build/main &&
pdflatex -output-directory=_build main && pdflatex -output-directory=_build main`. Output stays in `_build/`.
