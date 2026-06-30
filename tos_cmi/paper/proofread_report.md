# Proofreading pass report (pre-camera visual review)

Read-only audit (5 parallel agents: number-consistency, terminology, duplicate-claims, caption-overclaim,
citation-grammar) → central fixes. Scope: wording/consistency only — no new claims/results/numbers, no
method changes, honest-negative framing preserved. Both builds recompiled.

## Number-consistency (verified against Table 1 + result artifacts)
| Quantity | Value (authoritative) | Status |
|---|---|---|
| TSMNet / EEGNet latent dim | 210 / 16 | consistent everywhere |
| TSMNet subj decode ERM (MLP) → RZ(V_D) | 0.997 → 0.96 (≈ random 0.997) | consistent |
| TSMNet RZ linear / full 7-dim Fisher deletion | 0.93 / 0.92–0.98 | consistent |
| TSMNet task Z→RZ | 0.75 → 0.75 | consistent |
| EEGNet subj decode V_D: linear / MLP | 0.82→0.35 / 0.88→0.54 (random 0.73 / 0.81) | consistent |
| EEGNet task Z→RZ | 0.64 → 0.64 | consistent |
| selectivity EEGNet / TSMNet | 0.35–0.55 / 0.04–0.08 | consistent |
| nDcand TSMNet / EEGNet | ≈3 of 210 / ≈5 of 16 | consistent |
| EEGNet global-LPC subj decode endpoint | mean 0.1748 → **0.17** | **FIXED**: prose said 0.18 (mis-round); reconciled to data value 0.17 (Table+prose+caption) |
| EEGNet mean LOSO target | 0.43 → 0.39, paired-$t$ $p\le0.001$ at $\lambda\ge1$ | **FIXED**: unified notation (was "p<0.001"/no-scope in Table+caption) |
| collapse counts | Phase 2.0 = 26/27; Phase 2.1 = 9/9 | NOT stated in paper → no conflation risk (confirmed) |

## Fixes applied (wording/consistency only)
1. **EEGNet endpoint 0.18 → 0.17** (results body L77, Fig 5 caption) — data value is 0.1748; Table 1/CSV (0.17) were correct, prose was the outlier.
2. **p-value notation unified** to `$p\le0.001$ at $\lambda\ge1$` across results body, Fig 5 caption, and Table 1 (was `p<0.001` / no-scope variants).
3. **Off-by-one `\S` refs**: results Setup `\S2.1→\S2.2` (score-Fisher), `\S2.2→\S2.3` (direct-sum); method `\S4.1→\S4.2` (collapse diagnosis lives in the TSMNet results subsection).
4. **Caption/heading verb softened** (EEGNet, partial-removal C8): §4.3 heading and Fig 5 headline "removes leakage" → "linearly reduces leakage"; Table 1 decision "removes leakage" → "reduces (does not eliminate) leakage"; abstract "removes much more subject leakage" → "reduces much more subject decodability".
5. **Internal label removed**: results limitation note "Phase 3 establishes…" → "this study establishes…".
6. **Line-broken `Fig 2B`** in synthetic.tex → `Fig.~\ref{fig:synth}B` (regex had missed the newline split).
7. **ASCII quotes** in discussion.tex `"removes leakage"` → ` ``removes leakage'' `.
8. **De-dup**: results "Final claim." sentence (verbatim copy of the abstract closing) reworded to a non-duplicate "In sum, …".
9. **Caption symmetry**: added `($n{=}27$)` to Fig 5 caption to match Fig 4.

## What the audit confirmed already clean (no change)
- No forbidden success wording ("improves DG", "solves collapse", "certified/safe deletion on EEG", "certified accept", "always collapse", "generally low-rank removable").
- Both "de-domaining" uses are artifact-qualified ("apparent … artifact of the collapse").
- No 26/27 vs 9/9 conflation (neither count appears in the paper).
- Citations all `\citep`, no double-parentheses; 27 keys, **no new citations introduced**.
- Anonymity: clean (no identity strings in tex/bib/sections; figure-PDF metadata clean).

## Gate (both builds)
| | main.tex | tmlr_main.tex |
|---|---|---|
| compiles | yes | yes |
| pages | 12 | 12 |
| undefined cites / refs | 0 / 0 | 0 / 0 |
| overfull hbox | 0 | 0 |
| visible claim tags | 0 | 0 |
| forbidden wording | clean | clean |
| new citations | none (27 keys) | none |

## Deferred (not part of this text pass)
- Optional `\citet` rephrasings for a few appositive cites (audit rated non-required; left `\citep` to stay minimal).
- `.bib` numeric fields (volume/pages) still header-flagged "confirm before camera-ready".
- **Next pass = local PDF visual review** (no poppler here): TikZ Fig 1 readability, Table 1 width/wrapping, Fig 2–5 legibility, caption spacing, TMLR title block, page breaks.
