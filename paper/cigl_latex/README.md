# paper/cigl_latex/ — CIGL LaTeX-ready package (Phase 4F / v0.4)

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
`\citep{key}` is used where the entry's bibliographic identity is verified. A trailing `\todoverify`
(rendered as **[TODO: verify citation]**) marks entries whose venue/DOI/pages are still `% TODO` in the
`.bib`. Resolved (full DOI): MOABB, EEGNet, BNCI2014_001 (Tangermann 2012 + Brunner 2008), BNCI2015_001
(Faller 2012). Still TODO: Schirrmeister, DGCNN, RGNN, LGGNet, Li 2018, CCMI — see
`../cigl/CITATION_TODO_QUEUE.md`.

## To build later (only when authorized)
`pdflatex main && bibtex main && pdflatex main && pdflatex main` — **do not run in this phase.**
