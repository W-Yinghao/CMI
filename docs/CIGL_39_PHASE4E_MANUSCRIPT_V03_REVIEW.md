# CIGL_39 — Phase 4E Manuscript v0.3 Review

> Phase 4E summary (writing/analysis only; no GPU, no training, no experiments, no new dataset, no ablation,
> no λ-grid, no edge-CMI, no SOTA, no CITA/DualPC/Tri-CMI changes). Artifacts under `paper/cigl/`.

## Wording / artifact cleanup

- **Line-break hyphenation artifacts fixed** in `MANUSCRIPT_DRAFT.md`: `conditional-information`,
  `dynamic-edge`, `domain-adversarial` (were split across line wraps as `conditional-\ninformation`, etc.).
  A repo-wide scan of `paper/cigl/*.md` for `\w+-\n\s*\w+` now shows none of the flagged compounds.
- **Banned affirmative phrasings absent** as claims: "without harming source-task performance", "at no task
  cost", "eliminates leakage" (the literal strings now live only in the guard test; CLAIMS_AUDIT forbidden
  cells are paraphrased).

## Stale phase labels

Bumped to **Phase 4E / v0.3**: MANUSCRIPT_DRAFT (was 4B), RESULTS_TABLES (4C), STATISTICAL_SUMMARY (4D),
FIGURE_CAPTIONS (4D), CLAIMS_AUDIT (4B), README (4B), OPEN_PAPER_BLOCKERS (4B), TABLES_AND_FIGURES_PLAN (4B),
REFERENCES_VERIFIED (4C). RELATED_WORK_MATRIX keeps its historical "created Phase 4B; verification updated
Phase 4C/4E". No evidence numbers altered.

## Tables status

`RESULTS_TABLES_DRAFT.md` is paper-facing Markdown pipe tables T1–T5; T3/T4 combine the two datasets with
fold group, n_folds, ERM/reg source bAcc (mean [range] / mean [95% CI]), retention count, graph/node KL
reduction (mean, 95% CI, range), ERM-leakage / reg-reduces / target-guardrail counts, "reg leakage still
clears null", edge status, decision. No new numbers introduced in v0.3.

## CI status

Fold-level bootstrap (seed 0, 10 000 resamples, percentile) from existing per-fold JSON, unchanged from
Phase 4D. v0.3 adds an explicit statement that these are **descriptive across-fold variability, not
per-trial/per-seed inferential guarantees or a null test** (the null is the per-fold permutation audit).
Per-seed (folds×seeds) CI remains `TODO: finalize`.

## Citation TODO queue

New `paper/cigl/CITATION_TODO_QUEUE.md`: 10 items, each with current status / missing field / suggested
source / required-before-submission. **Dataset primaries (BNCI2014_001 2a; BNCI2015_001 Faller 2012) are the
highest-priority opens and are required before submission.** Items 3–6, 9–10 need exact vol/DOI; items 7–8
(RGNN/LGGNet) are arXiv-citable now. Nothing fabricated.

## Figure captions

`FIGURE_CAPTIONS_DRAFT.md` (F1–F4) polished: F2 reframed as leakage-reduction-vs-retention-gate (retention
axis, not accuracy gain); F4 reframed so negatives read as deliberate scoping. Each caption keeps visual /
supports / does-not-support / source. No assets generated.

## Remaining blockers (all writing/analysis — no GPU)

1. Dataset-primary citations (required before submission) + remaining vol/DOI fields.
2. Prose pass for §3/§4 at submission length; LaTeX conversion of `[TABLE]`/`[FIGURE]` placeholders.
3. Generate F1–F4 assets (captions ready).
4. Per-seed CI finalization for camera-ready.

## Is any experiment paper-critical?

**No.** Bounded claim fully supported by CIGL_25/29/31; `graph_010` vs `node_010` mechanism ablation stays
optional (see `ABLATION_DECISION_MEMO.md`).

## Adversarial verification (pre-push)

A 5-dimension adversarial workflow (8 agents; each fix_now finding re-checked by a refute-by-default
skeptic) audited v0.3 before push:
- **drift**: ZERO numeric/claim drift vs v0.2 — every value preserved (35–58%, 31–45%, 43–77%, 37–61%; fold9
  +0.024, 11/12; 8×/15×, p=0.020, corr 0.945); bounded disclaimers intact. Safe to push.
- **artifacts**: clean (no hyphenation, no stale current-state labels, no banned affirmative phrasings).
- **paper-critical**: the paper stands **without any new experiment**; the `graph_010` vs `node_010`
  mechanism ablation stays optional.
- **overclaim**: one medium item — the fold9 retention miss was only in §5 while CLAIMS_AUDIT row 12 wants it
  in a main claim. **Fixed**: C3 now states "the gate holds dataset-wide, with one BNCI2015_001 fold missing
  the per-fold threshold."
- **citations**: the only confirmed fix_now items all reduce to "citations remain TODO and are required
  before submission" — already catalogued in `CITATION_TODO_QUEUE.md` / `OPEN_PAPER_BLOCKERS.md` and **left
  TODO, not fabricated** (per the reviewer's instruction). Honesty polish applied: `REFERENCES_VERIFIED.md`
  now defines "verified" = title/venue/year baseline, and the DGCNN venue descriptor reads "IEEE (venue TBD)".

## Confirmations

- No GPU / no training / no new dataset / no ablation / no λ-grid; no edge-CMI; no SOTA framing; no
  CITA/DualPC/Tri-CMI changes; no generated tables committed (gitignored).
- Validation: py_compile OK; collector + manuscript-claims tests pass (incl. new artifact guards); collect
  dry-run OK; reviewer paper-facing wording-check block passes.

## Recommendation (pending reviewer)

**A — v0.3 ready for a human prose pass / LaTeX conversion.** No experiment needed; open items are
citations (esp. dataset primaries), a prose pass, and figure-asset generation. Authorize the `graph_010` vs
`node_010` ablation only if a human review finds the per-term mechanism essential.
