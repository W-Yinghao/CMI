# CIGL Manuscript v0.4 — Review Notes (Phase 4F)

> Self-review of the v0.3 → v0.4 transition (citation closure + LaTeX-ready package; writing/packaging
> only). Externally-facing summary: `docs/CIGL_40_PHASE4F_LATEX_READY_REVIEW.md`.

## What changed from v0.3

- **Citation closure (4/10 resolved).** Reviewer-verified full fields added for BNCI2014_001 (Tangermann
  et al. 2012, DOI 10.3389/fnins.2012.00055, + Brunner et al. 2008 Graz-2a description), BNCI2015_001
  (Faller et al. 2012, IEEE TNSRE 20(3):313–319, DOI 10.1109/TNSRE.2012.2189584), MOABB (Jayaram &
  Barachant, JNE 15(6):066011, DOI 10.1088/1741-2552/aadea0), EEGNet (Lawhern et al., JNE 15(5):056013,
  DOI 10.1088/1741-2552/aace8c). The **dataset-primary blocker is cleared**.
- **Draft BibTeX** `REFERENCES_DRAFT.bib`: verified-identity entries only; unverified venue/DOI/pages are
  inline `% TODO` comments (no fabricated values); header warns it is draft.
- **LaTeX-ready package** `paper/cigl_latex/`: `main.tex` (neutral `article` class, no conference template,
  no PDF), 7 section `.tex` files (faithful conversion of v0.3 prose, claims unchanged), 5 table `.tex`
  drafts (existing values + CIs), `figures/FIGURE_ASSET_PLAN.md`, `README.md`.
- `REFERENCES_VERIFIED.md` / `CITATION_TODO_QUEUE.md` updated to mark the 4 resolved items.

## Citation TODOs — resolved vs remaining

- **Resolved (4):** BNCI2014_001, BNCI2015_001, MOABB, EEGNet (full DOIs).
- **Remaining (6):** Schirrmeister (vol/pages/DOI), DGCNN (exact venue/year/DOI), Li 2018 (author
  list/DOI), CCMI (vol/pages) — required before submission; RGNN, LGGNet — arXiv-citable now, upgrade to
  published if confirmed. None fabricated; all visible as `\todoverify` in the `.tex` and `% TODO` in `.bib`.

## LaTeX package status

Structurally complete and self-consistent (main `\input`s all sections; tables match
`RESULTS_TABLES_DRAFT.md`/`STATISTICAL_SUMMARY_DRAFT.md`). **Not compiled** (no PDF). No conference style
added (per scope). Section text preserves every bounded claim; a claim-fidelity audit found zero drift.

## Table status

T1–T5 as LaTeX `tabular` drafts; T3/T4 carry the fold-level bootstrap CIs (seed 0). No new numbers.

## Figure status

F1–F4 are plans + captions only; **no images generated**. F2/F3 (bars/scatter) are auto-generatable from
existing JSON later; F1/F4 are hand-drawn schematics. No new experiments implied.

## Remaining blockers before submission (all writing/packaging — no GPU)

1. 6 remaining citation fields (esp. Schirrmeister/DGCNN/Li/CCMI exact venue/DOI/pages).
2. Generate F1–F4 assets (F2/F3 from existing JSON; F1/F4 hand-drawn).
3. Human prose pass + a conference template/style when the venue is chosen.
4. Authorized PDF compile.

## Is any experiment paper-critical?

**No.** The bounded claim is fully supported by CIGL_25/29/31; the `graph_010` vs `node_010` mechanism
ablation remains optional (`ABLATION_DECISION_MEMO.md`).

## Recommendation

**A — ready for human prose / LaTeX editing.** Open items are citation-field finalization, figure assets,
and a venue template; no experiment is required.
