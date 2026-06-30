# CIGL Page-Budget Analysis (Phase 4I)

> Planning only. Counts are from the v0.6 source; the "compressed" estimates are projections, not compiled
> measurements under a venue template.

## Current review build

- Template: neutral `article`, 11pt, single-column. **Review PDF = 12 pages** (`_build/main.pdf`).
- The 12 pages are **float-dominated**: prose is short (~1.9k words), so most pages are tables/figures and
  float whitespace, not dense text.

## Length by section (prose words; tables/figures separate)

| section | words | notes |
|---|---|---|
| Abstract | ~210 | within typical 150–250 limit |
| §1 Introduction (+contributions) | ~376 | could trim ~10% |
| §2 Related Work | ~119 | **short** — likely needs expansion for a journal/AAAI |
| §3 Method | ~368 | core; do not cut |
| §4 Protocol | ~117 | concise; fine |
| §5 Results | ~269 | core; defers detail to T3/T4 |
| §6 Analysis + Negative Results | ~227 | core; keep |
| §7 Limitations + Conclusion | ~209 | core; keep |
| **Total prose** | **~1.9k words** | a *concise* paper — the risk is under-length narrative, not over-length |

Floats: **5 tables** (T1 method, T2 audit, T3 BNCI2014_001, T4 BNCI2015_001, T5 negative results),
**4 figures** (F1 pipeline, F2 reduction-vs-retention, F3 audit-vs-null, F4 decision flow), **11 references**.

## Projected compressed length

- **AAAI-like two-column (dense), ~7–8 pp limit:** the ~1.9k words + 5 tables + 4 figures + refs project to
  roughly **5–7 two-column pages** — **fits** the limit, with **headroom to expand** §2 and discussion. The
  binding constraint would be *float real estate* (9 floats is a lot for 7 pp), not text.
- **TMLR single-column:** **no hard limit**; current content is comfortable; expand discussion as desired.
- **Workshop (4–8 pp):** fits; may need to move T3 *or* T4 detail + F3 to an appendix.

## If page-limited: move to appendix (in priority order)

1. One of the per-dataset confirmation tables' *full* detail (keep a combined headline T3/T4 row in main text).
2. F3 (audit-vs-null) detail and the per-fold scatter inset of F2.
3. Extended related-work discussion / extra citation context.
4. The exact permutation-null + bootstrap-CI methodology paragraphs (summarize in main, full in appendix).

## Must NOT be cut (load-bearing)

- **Method definition** (§3: posterior-KL proxy, two-step regularizer, audit/null).
- **Two-dataset confirmation headline** (the partial-reduction-at-retention result on BNCI2014_001 +
  BNCI2015_001, with the fold9 caveat).
- **Negative results** (§6 / T5) — they justify the bounded scope; cutting them would make the scope look
  arbitrary.
- **Limitations** (§7) — proxy/partial/one-backbone/two-datasets/one-λ; central to the honesty of the claim.

## Headline risk

The paper is **concise**, not over-length. For AAAI/journal, the more likely reviewer reaction is "thin /
narrow," so the page budget should be spent **expanding §2 and the discussion of the measurement→control
gap**, not cutting. For a workshop, the current length is already close to right.
