# CIGL Manuscript v0.3 — Review Notes (Phase 4E)

> Self-review of the v0.2 → v0.3 transition (writing/analysis only). For the externally-facing summary see
> `docs/CIGL_39_PHASE4E_MANUSCRIPT_V03_REVIEW.md`.

## What changed from v0.2

- **Hyphenation artifacts fixed.** The three line-break hyphenations (`conditional-information`,
  `dynamic-edge`, `domain-adversarial`) are joined so Markdown/LaTeX conversion cannot mangle them.
- **Stale phase labels bumped.** `MANUSCRIPT_DRAFT` (4B→v0.3/4E), `RESULTS_TABLES` (4C→4E),
  `STATISTICAL_SUMMARY`/`FIGURE_CAPTIONS` (4D→4E), `CLAIMS_AUDIT`/`README`/`OPEN_PAPER_BLOCKERS`/
  `TABLES_AND_FIGURES_PLAN` (4B→4E), `REFERENCES_VERIFIED` (4C→4E); `RELATED_WORK_MATRIX` keeps its
  historical "created 4B; verification updated 4C/4E".
- **Introduction tightened** into a gap→approach paragraph that no longer duplicates the C1–C4 list; same
  bounded claims, fewer repeated sentences — reads as a paper intro, not a project log.
- **Results de-numbered** in prose: per-dataset reductions now defer to T3/T4 (with CIs) instead of
  repeating every range; retention wording is consistently gate-based ("meets the retention gate"), and the
  fold9 miss is stated inline.
- **Statistical summary** now states explicitly that the fold-level bootstrap CIs are descriptive
  across-fold variability, **not** per-trial/per-seed inferential guarantees or a null test.
- **Figure captions**: F2 reframed as leakage-reduction-vs-retention-gate (a retention axis, not accuracy
  gain); F4 reframed so the negatives read as deliberate scoping, not failures.
- **New docs**: `CITATION_TODO_QUEUE.md` (per-citation queue), this file, `docs/CIGL_39`.
- **Tests** strengthened to reject the three line-break hyphenations plus the previously-banned phrases.

## Remaining writing blockers (no GPU)

1. Prose pass for §3/§4 flow at submission length (method/protocol are precise but dense).
2. Convert `[TABLE]`/`[FIGURE]` placeholders to real refs at LaTeX time.
3. Final consistency sweep of numbers vs T2/T3/T4 at camera-ready.

## Citation blockers

- **Dataset primaries (items 1–2 in `CITATION_TODO_QUEUE.md`) are required before submission** and still
  open. Items 3–6, 9–10 need exact vol/DOI; items 7–8 are arXiv-citable now.

## Figure blockers

- F1–F4 are caption-only; assets not generated (out of scope for a writing phase). No figure asserts a
  claim beyond CIGL_25/29/31.

## Is any experiment paper-critical?

**No.** The bounded claim (audit + partial task-preserving reduction, two MI datasets, graph/node-only,
fixed λ) is fully supported by CIGL_25/29/31. The `graph_010` vs `node_010` mechanism ablation remains
**optional** under the paired-penalty framing (see `ABLATION_DECISION_MEMO.md`); it would add a per-term
mechanism statement, not rescue the main claim.

## Recommendation

**A — v0.3 ready for a human prose pass / LaTeX conversion.** No experiment is needed; the open items are
citations (esp. dataset primaries), a prose pass, and figure-asset generation.
