# CIGL Venue Decision Memo (Phase 4I)

> Planning only. No deadlines are asserted (verify current CFP dates separately before committing). The goal
> is to pick the venue whose review culture best fits a **bounded, honestly-scoped, negative-results-forward
> measurement→control paper** — not a leaderboard paper.

## What kind of paper this is

- **Contribution type:** a *measurement-and-partial-control* result with explicit scope and negative results.
  Not SOTA, not a new benchmark, not a large-scale empirical study.
- **Strengths reviewers should value:** source-only firewall, retrained permutation null, honest "partial,
  not elimination" framing, two-dataset confirmation of a *frozen* candidate, and the negative results that
  shaped the scope.
- **Weaknesses reviewers will probe:** single backbone, two MI datasets, one fixed λ, a *proxy* (not
  unbiased CMI), modest baselines, no per-term (graph-only vs node-only) mechanism.

## Candidate venues

| venue | fit | page budget | main risk | notes |
|---|---|---|---|---|
| **TMLR** | **strong** | flexible (no hard limit) | "is the scope too narrow?" — TMLR judges correctness+claims, not novelty/SOTA, which matches our bounded story | **Recommended first.** Rewards honest scope + negative results; no SOTA pressure; rolling submission. |
| **ML4H / NeurIPS or ICLR workshop** | strong (early venue) | short (4–8 pp) | page budget tight; archival status varies | Good for fast feedback; health/EEG audience understands leakage. Non-archival ones keep options open. |
| **IEEE TNSRE / J. Neural Engineering** | medium–strong | journal-length | expects clinical/BCI utility framing + stronger baselines | EEG-native audience; would want the leakage-control framed for BCI deployment; longer review. |
| **AAAI (main track)** | medium | ~7–8 pp + refs | novelty/SOTA expectations; single-backbone/two-dataset scope may read as thin for a main-track AI paper | Possible but higher rejection risk for a deliberately-bounded paper; template is heavy. |

## Fit analysis

- **TMLR** is the best match: its acceptance criteria are "claims supported by evidence" and "of interest to
  some audience," explicitly *not* novelty or beating SOTA. Our bounded claim + permutation null + honest
  limitations are exactly what TMLR rewards, and the flexible length lets the negative results and
  limitations breathe.
- **A workshop** (ML4H / NeurIPS / ICLR) is the best *fast* option for feedback and to socialize the
  measurement→control gap; pick a non-archival one if we may extend to TMLR/journal later.
- **AAAI main track** is feasible but riskier: the deliberately-narrow scope and lack of SOTA invite
  "insufficient contribution" reviews; would need careful framing of the *measurement* novelty.

## Template / formatting to check (before migrating)

- TMLR: official `tmlr.sty` two-column? (No — TMLR is single-column LaTeX style); confirm current style file
  and anonymization rules.
- AAAI: `aaai24`/current `aaai`-style `.sty` + strict page limit incl. references handling; `\pdfinfo` rules.
- Workshop: per-workshop style (often NeurIPS/ICLR style); confirm archival status.
- All: anonymization (remove author/affiliation; our `main.tex` already uses an anonymous author line).

## Recommendation

**Target TMLR first** (best culture fit for a bounded, honest, negative-results paper), with a **workshop
submission in parallel for fast feedback** if a suitable non-archival one is open. Hold AAAI/journal as
fallbacks. Do **not** migrate to a venue template until the human prose pass is done and the venue is chosen;
the current neutral `article` build is correct for human reading now.

`TODO before committing to a venue:` verify the current CFP/deadline and the exact, current style file for
the chosen venue.
