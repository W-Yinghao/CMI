# CIGL — TMLR Preflight Notes (Phase 4J handoff)

> **Documentation only — no template migration is performed here.** These are the items to check/prepare
> *before* migrating the neutral `article` build to the TMLR stylefile. Verify each against the **current**
> TMLR author guide at submission time (policies/stylefiles change).

## What TMLR requires (to confirm against the current author guide)

- **Double-blind review** — the submission PDF must be anonymized (no author names/affiliations,
  no identifying acknowledgements, no self-identifying links/repo names).
- **TMLR LaTeX stylefile** — the submission PDF must be produced with the official TMLR style; the current
  neutral `article` build is **not** the final format. (`TODO: fetch the current `tmlr.sty`/`.cls` and the
  exact class options at migration time.)
- **Code/data availability statement** — typically expected; prepare a short statement (what is released,
  how to reproduce F2/F3 from existing JSON, the source-only firewall).
- **Conflicts / author profiles / OpenReview** — TMLR runs on OpenReview; authors complete profiles and
  declare conflicts. (`TODO: confirm at submission`.)
- **Supplementary material policy** — confirm what may go in an appendix vs separate supplement (relevant to
  the page-budget plan: per-fold table detail, F3 node-map, methodology).
- **Broader impact / ethics** — include if applicable (EEG/clinical data; subject privacy is directly
  relevant given the leakage topic).

## Anonymization checklist (before building the TMLR PDF)

- `main.tex` author line is already anonymous ("authors omitted for anonymous draft") — keep it.
- Strip identifying strings from text **and figures**: internal doc names (`CIGL_NN`), branch names,
  datalake/cluster paths, any institution-specific wording.
- Remove acknowledgements/funding until camera-ready.
- Ensure committed figure assets (`fig*_draft.svg/.png`) contain no identifying metadata.

## Migration steps (run only after PI prose read + reviewer authorization)

1. Add the current `tmlr` stylefile to `paper/cigl_latex/` (or a new `paper/cigl_tmlr/`); set
   `\documentclass` + required packages; keep `\input` section/table structure.
2. Re-point `\bibliography` to `REFERENCES_DRAFT.bib`; convert to the TMLR bib style.
3. Replace placeholder F1/F4 floats with TikZ/PDF (see `FIGURE_FINALIZATION_PLAN.md`).
4. Rebuild review PDF into gitignored `_build/`; check page budget under the TMLR single-column style.
5. Run anonymization sweep; add code/data availability + (if needed) broader-impact statements.
6. Final claims/citation/figure honesty pass (reuse the guard tests).

## Do NOT do yet

- Do not migrate the template, change `\documentclass`, or commit a TMLR-styled PDF before the PI prose read
  and reviewer authorization. Template-before-prose creates formatting churn.
