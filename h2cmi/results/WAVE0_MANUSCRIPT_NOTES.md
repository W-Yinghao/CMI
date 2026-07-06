# Wave 0 — manuscript-ready inserts (stage for when the h2cmi paper is drafted)

No h2cmi LaTeX manuscript exists yet; these are the exact inserts to use, per the W0.1 result. Numbers are
from `wave0_w2.report.json` (deterministic eval-only re-eval, analyzer `9a35cc9`-equivalent on branch
`exp/h2cmi-wave0-mechanism`), self-replay 27/27 bit-identical.

## Insert 1 — Results / Sleep staging (mechanism)

> A deterministic W0.1 rerun saved branch-level predictions and passed 27/27 bit-identical self-replay.
> The four-decimal terminal decomposition was unchanged: G = −0.0201, P = −0.1439, and fixed-prior
> iterative minus joint-fit geometry = +0.0187. The newly admissible per-stage confusion localizes the
> decision-prior harm: using the fitted prior as the decision prior collapses minority-stage recall,
> e.g. N1 0.288 → 0.006 and REM 0.639 → 0.307.

## Insert 2 — Discussion / Limitations (REPLACES the old "W2 confusion not saved / should save in future")

> The earlier branch-level confusion limitation was addressed by a deterministic replay-locked W0.1 run.
> Remaining limitations are external-cohort coverage and the restriction to the frozen latent diagonal
> family.

## Insert 3 — Figure / Table (per-stage recall across the four branches)

Full 5-stage per-stage recall (from `wave0_w2.report.json`, primary protocol, n=75 subjects). Compact
inset for the main text = the N1 + REM columns; full table → appendix.

| branch | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| identity, Unif       (`I,  Unif`) | 0.924 | **0.288** | 0.885 | 0.534 | **0.639** |
| identity, π_J        (`I,  π_J`)  | 0.952 | **0.006** | 0.817 | 0.483 | **0.307** |
| joint-geom, Unif     (`T_J,Unif`) | 0.924 | 0.213 | 0.499 | 0.962 | 0.640 |
| fixed-ref one-shot, Unif          | 0.907 | 0.316 | 0.629 | 0.932 | 0.732 |

Reading: the `(I, π_J)` row vs `(I, Unif)` row isolates the **decision-prior** effect at fixed geometry —
minority stages N1/REM collapse. `fixed-ref one-shot` keeps the most balanced recall.

## Supersession (manuscript bookkeeping)

- OLD: "W2 primary rows did not save complete branch-level confusion; replay failed hash-equivalence, so
  replay-derived per-stage recall/confusion are excluded." → **SUPERSEDED** by W0.1 (deterministic,
  self-replay-locked; confusion admissible). Keep a one-line reproducibility-audit note pointing to W0.1.
- The `278fc85` / `h2cmi-review-p0-terminal` scalar decomposition is unchanged (re-confirmed to 4 dp);
  W0.1 adds the mechanism layer, it does not revise the headline numbers.
