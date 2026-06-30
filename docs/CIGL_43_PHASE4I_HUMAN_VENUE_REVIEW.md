# CIGL_43 — Phase 4I Human Reading Pass + Venue Planning Review

> Phase 4I summary (writing/planning only; no GPU, no training, no experiments, no ablation, no new dataset,
> no λ-grid, no edge-CMI, no SOTA, no CITA/DualPC/Tri-CMI changes, no PDF/aux committed, no invented
> citations). Follows the Phase 4H metadata patch (905d6e6, branch merge-ready as v0.6).

## Prose pass (readability only; claims unchanged)

De-project-logged the section prose: replaced internal phase codes and raw flag names with descriptive
wording — `graph_node_010` → "our fixed graph/node regularizer ($\lambda_g=\lambda_n=0.010$)";
`confirmed_with_target_guardrail = true` → "passing the full source-only confirmation including the target
guardrail"; the F4 caption no longer uses `3A-R/3A-S/3A-G/3A-H/3A-I/J/K`; "3A-H audit" → "development-fold
audit". No claims, numbers, tables, figures, or citations changed.

## Planning docs created

- `paper/cigl_latex/VENUE_DECISION_MEMO.md` — candidate venues + fit; **recommend TMLR first** (rewards
  bounded, honest, negative-results paper; no SOTA pressure), workshop in parallel for fast feedback; AAAI/
  journal as fallbacks. CFP/deadline = verify-before-committing.
- `paper/cigl_latex/PAGE_BUDGET_ANALYSIS.md` — current 12-pp `article` build is float-dominated; prose is
  ~1.9k words (concise). Projects to ~5–7 two-column pp under an AAAI-like template (**fits**, with headroom
  to expand §2/discussion). Load-bearing (never cut): method, two-dataset confirmation, negative results,
  limitations. Headline risk is *under-length narrative*, not over-length.
- `paper/cigl_latex/FIGURE_FINALIZATION_PLAN.md` — F1/F4 → TikZ (remove internal codes); F2/F3 regenerate
  from existing JSON/audit values; cross-cutting honesty checks.
- `paper/cigl_latex/SUBMISSION_READINESS_CHECKLIST.md` — claims/citation/table/figure/firewall/artifact/
  anonymization/reproducibility status + remaining (writing) blockers.

## Compile / tests

Neutral `article` review build still compiles; the prose edits are removals of `\texttt{...}` flag names and
codes (no structural change). Tests (`test_cigl_latex_package`, `test_cigl_manuscript_claims`) pass; planning
docs present (≥20 lines each). No generated PDF/aux committed.

## Is any experiment paper-critical?

**No.** The bounded paper stands; `graph_010` vs `node_010` ablation remains optional and unauthorized.

## Recommendation

**B then A:** a **human PI prose edit** (tone + §2 expansion) is the most valuable next step, after which
**proceed to venue-template migration** (TMLR or chosen venue). No experiment is required.
