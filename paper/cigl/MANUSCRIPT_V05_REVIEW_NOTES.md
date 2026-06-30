# CIGL Manuscript v0.5 — Review Notes (Phase 4G)

> Self-review of the v0.4 → v0.5 transition (citation closure + LaTeX compile smoke; writing/packaging
> only). Externally-facing summary: `docs/CIGL_41_PHASE4G_CITATION_AND_COMPILE_REVIEW.md`.

## Citation TODOs — resolved vs remaining

- **All 10 citations RESOLVED.** Phase 4G added full fields for the 6 that were open:
  - Schirrmeister → *HBM* 38(11):5391–5420, 2017, DOI 10.1002/hbm.23730 (reviewer-verified).
  - DGCNN (Song et al.) → *IEEE TAC* 11(3):532–541, 2020 (early access 2018), DOI 10.1109/TAFFC.2018.2817622
    (reviewer-verified).
  - Li et al. → *Proc. AAAI* 32(1), 2018, DOI 10.1609/aaai.v32i1.11682 (reviewer-verified).
  - CCMI → *PMLR* 115:1083–1093 (UAI), 2020 (reviewer-verified; PMLR has no DOI).
  - **RGNN → *IEEE TAC* 13(3):1290–1301, 2022, DOI 10.1109/TAFFC.2020.2994159 — independently
    Crossref-verified** (reviewer had flagged it "likely").
  - **LGGNet → *IEEE TNNLS* 35(7):9773–9786, 2024, DOI 10.1109/TNNLS.2023.3236635 — Crossref-verified;
    the 3rd author was corrected from "Zhang" to "Tong"** (Chengxuan Tong), and the year/venue corrected to
    the published 2024 TNNLS version.
- **Only remaining (minor, not a citation):** the Brunner Graz-2a *data-description URL* (a `note`/comment,
  not required for a valid citation). All `\todoverify` markers were removed from the `.tex` since no
  citation key is unresolved.

## Compile smoke result

**SUCCESS.** `pdflatex → bibtex → pdflatex ×2` (no `latexmk` installed) compiles `main.tex` end-to-end:
all passes exit 0, BibTeX reports **11 entries, no errors**, **no undefined citations/references**, output
`_build/main.pdf` (~318 KB, **8 pages**). One fix was needed during the smoke: two `% TODO` comments were
moved out of their BibTeX entry bodies (BibTeX disallows in-entry `%` comments). Details in
`paper/cigl_latex/COMPILE_SMOKE_SUMMARY.md`. **No PDF/aux/bbl/log committed** (`_build/` + LaTeX artifacts
gitignored; `*.pdf` already global-ignored).

## Is the LaTeX package ready for human editing?

**Yes.** It compiles cleanly with a complete bibliography and bounded claims unchanged (claim-fidelity was
audited in Phase 4F and the prose is unchanged here). A human can now do prose editing and choose a venue
template.

## Remaining paper blockers (all writing/packaging — no GPU)

1. Minor: Brunner Graz-2a data-description URL (a note field).
2. Generate F1–F4 figure assets (F2/F3 from existing JSON; F1/F4 hand-drawn).
3. Human prose pass; add a conference template/style once the venue is chosen.
4. An authorized full PDF build for submission.

## Is any experiment paper-critical?

**No.** Bounded claim fully supported by CIGL_25/29/31; `graph_010` vs `node_010` mechanism ablation remains
optional (`ABLATION_DECISION_MEMO.md`).

## Recommendation

**A — ready for human LaTeX prose editing.** Citation and compile blockers are cleared; no experiment is
required.
