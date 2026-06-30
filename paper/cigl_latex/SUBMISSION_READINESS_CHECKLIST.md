# CIGL Submission-Readiness Checklist (Phase 4I)

> Status of each gate to submission. ✅ done · ◻ pending (writing/packaging) · ⚠ needs reviewer/human.
> No item here implies a new experiment.

## Claims audit
- ✅ Bounded claim fixed: posterior-KL proxy (not unbiased CMI); partial graph/node reduction (not
  elimination); source-task retention criteria (gate-based, fold9 disclosed); graph/node only; edge-CMI out
  of scope; dynamic-edge unsupported (consistent-with, not causal); no SOTA; no beyond-MI; no λ-robustness.
- ✅ Guard test `tests/test_cigl_manuscript_claims.py` + `tests/test_cigl_latex_package.py` enforce wording.
- ⚠ Human PI read-through to confirm tone/over-claim once more before submission.

## Citation audit
- ✅ 10/10 references resolved with verified fields (`REFERENCES_DRAFT.bib`); RGNN/LGGNet + 3 reviewer DOIs
  Crossref-verified.
- ◻ Minor: Brunner Graz-2a data-description URL (a `note`).
- ◻ Convert `REFERENCES_DRAFT.bib` → final `.bib` at venue-template time (no field changes expected).

## Table audit
- ✅ T1–T5 rendered (all `\input`), numbers match CIGL_29/31/25 (number-fidelity verified).
- ◻ Decide which tables move to appendix if page-limited (see `PAGE_BUDGET_ANALYSIS.md`).

## Figure audit
- ✅ F2/F3 data-backed (real JSON/audit values), embedded; F1/F4 schematic drafts (placeholder floats).
- ◻ F1/F4 → TikZ; remove internal codes; 300dpi (see `FIGURE_FINALIZATION_PLAN.md`).
- ✅ No figure implies unobserved data (self-verified).

## Source-only firewall statement
- ✅ Stated in §4 and abstract: **target labels are evaluation-only** — never used for training, early
  stopping, normalization, model/config selection, confirmation-label choice, probe fitting, or the audit.
- ✅ Firewall + target-corruption tests exist in the codebase (CIGL_36); cite in a reproducibility appendix.

## Generated-artifact policy
- ✅ No PDF/aux/bbl/blg/log committed; `_build/` + LaTeX artifacts gitignored; `*.pdf` globally ignored.
- ✅ Committed figure assets are source `.svg`/`.png` only.
- ✅ `git ls-files` shows no tracked LaTeX build artifacts (tested).

## Anonymization
- ✅ `main.tex` uses an anonymous author line ("authors omitted for anonymous draft").
- ◻ Strip identifying repo paths / internal doc names (CIGL\_NN, branch names, datalake paths) from the
  camera-ready text and figures.
- ◻ Remove acknowledgements / funding until camera-ready (per venue double-blind rules).

## Reproducibility
- ◻ Add a reproducibility appendix: backbone (DGCNN static adjacency), λ=0.010, seeds 0–2, n\_perm=50,
  gate α=0.05, MOABB preprocessing (128 Hz, 0.5–3.5 s, per-trial z-score), datalake provenance.
- ◻ Ship the F2/F3 figure-generation script (values-only from existing JSON).
- ◻ State that no target labels were used outside evaluation (firewall).

## Remaining blockers (all writing/packaging — no GPU)
1. Human PI prose pass (tone, §2 expansion, transitions).
2. Venue choice + template migration (see `VENUE_DECISION_MEMO.md`); verify current CFP/style.
3. F1/F4 TikZ finalization; remove internal codes everywhere.
4. Minor Brunner URL; final `.bib`; reproducibility + anonymization sweeps.
5. Authorized full submission-PDF build under the chosen template.

## Not blockers (explicitly out of scope)
- graph-only vs node-only ablation (optional; unauthorized), third dataset, λ-grid, edge-CMI, SOTA table.
