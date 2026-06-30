# CIGL — PI Reading Guide (Phase 4J handoff)

> A 10-minute guide for the PI to read the current v0.6 draft efficiently and give prose feedback. **No
> manuscript content was changed in this packet** — it only orients the read. Hold all template migration
> until after this read.

## Read in this order (≈ the review PDF)

1. **Abstract** (`main.tex`) — does the bounded claim land in one breath?
2. **§1 Introduction + Contributions** (`sections/01_intro.tex`) — is the measurement→control gap compelling?
3. **§5 Results** + **Tables 3–4** + **Figure 2** — is the partial-reduction-at-retention result persuasive
   *without* SOTA framing?
4. **§6 Analysis & Negative Results** + **Table 5** + **Figure 4** — do the negatives read as *evidence*?
5. **§3 Method**, **§4 Protocol** — does it read like a method, not a project log?
6. **§7 Limitations & Conclusion** — honest but not self-defeating?
7. **§2 Related Work** — the thinnest section; likely needs expansion before TMLR.

(Build the review PDF locally: from `paper/cigl_latex/`, `pdflatex -output-directory=_build main` →
`bibtex _build/main` → `pdflatex` ×2. `_build/` is gitignored; ~12 pages.)

## Intended one-sentence contribution

> On a *task-capable* EEG graph backbone, label-conditional domain leakage in the graph/node representations
> is **measurable** (source-only, permutation-null audit) and **partially controllable** (a fixed graph/node
> posterior-KL regularizer reduces it while meeting a pre-registered source-task retention gate on two MI
> datasets) — a measurement→control result, not a leaderboard result.

## The exact bounded claim (do not let prose drift past this)

- posterior-KL **proxy**, explicitly **not** an unbiased CMI estimator;
- **graph/node only**; static adjacency; **no edge object / edge-CMI out of scope**;
- single DGCNN static-adjacency backbone; **two MI datasets** (BNCI2014_001, BNCI2015_001);
- strict **source-only** DG; **target labels evaluation-only**;
- **partial** reduction (regularized leakage still clears the null) — **not** elimination;
- single **fixed** λ = 0.010 (no λ-grid, no λ-robustness claim);
- **one BNCI2015_001 fold (fold9) misses the per-fold retention threshold**; the dataset-level gate passes.

## Claims that must NOT appear

SOTA / best accuracy; leakage elimination / removal; unbiased CMI / information-theoretic guarantee; a
working edge-CMI or dynamic-edge method; cross-architecture or beyond-MI generality; "no task cost" /
"without harming performance"; any causal claim that the per-sample adjacency `A(x)` *is* the leakage source.

## High-risk paragraphs to inspect

- **Abstract** and **§1** — most likely place a strengthened claim sneaks in; check every adjective.
- **§5 "Fixed regularizer…"** — ensure retention is *gate-based* and the fold9 miss is visible.
- **§6 (iii) dynamic-edge** — must stay "consistent with a fingerprint channel, **not** a causal isolation."
- **§3 leakage proxy** — must stay "proxy, not unbiased CMI."

## Questions the PI should answer (feeds the next prose edit)

1. Story understandable without the phase history? 2. Intro gap compelling? 3. Related Work expansion needed?
4. Negatives as evidence, not excuses? 5. Method reads as a method? 6. Results persuasive without SOTA?
7. Limitations honest but not self-defeating? 8. Is TMLR the right first target?

## Current recommendation

**TMLR first** (review culture fits a bounded, honest, negative-results paper; no SOTA pressure), with a
**workshop in parallel** for fast feedback if a suitable non-archival one is open. Migrate to a venue
template **only after** this prose read (see `TMLR_PREFLIGHT_NOTES.md`).
