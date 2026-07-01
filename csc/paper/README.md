# CSC manuscript (`csc/paper/`)

**Title.** *When Can EEG Concept Shift Be Certified? From Z-only Abstention to Paired Minimal-Label
Certificates.*

**Status.** Skeleton. Claims, scope, numbers, and tables are locked to committed artifacts (see
`claim_evidence_table.md`); prose is stubbed with visible `[TODO: ...]` markers. This is not a finished
paper.

## The story (5-part main line)
1. **Theory** — Z-only non-identifiability of concept shift; abstention is necessary (`sections/theory.tex`, Prop. 1).
2. **Route A audit** — source-anchored Z-only certificate; frozen confirmatory FAIL on unseen clusters (`sections/routeA.tex`).
3. **Route B3 method** — paired within-subject target-internal reference + minimal labels (`sections/routeB3_method.tex`).
4. **Route B3 confirmatory** — frozen unseen synthetic PASS (C1–C6) within a scoped envelope (`sections/routeB3_confirmatory.tex`).
5. **Boundary** — synthetic-only, envelope limits, pure-conditional tail, real-EEG as future work (`sections/discussion.tex`).

## Build

```bash
bash csc/paper/compile.sh        # neutral build -> main.pdf (self-contained, base LaTeX)
bash csc/paper/check_claims.sh   # forbidden-wording + required-scope + compile gate
```

### Two-build convention (venue build — OPEN DECISION)
This mirrors `tos_cmi/paper` (neutral `main.tex` + an official-stylefile variant). The **venue is not yet
chosen**, so only the neutral build is wired. To add the official build once the venue is fixed:
1. drop the venue style/bst files into this directory (e.g. `aaai/neurips/tmlr`);
2. add `venue_main.tex` = `main.tex` with the venue `\documentclass`/`\usepackage`, `\showtodofalse`, and
   anonymization for the camera/submission variant;
3. add `compile_venue.sh` (same 3-pass `pdflatex`/`bibtex` sequence).
Do **not** hardcode a venue in `main.tex`.

## Draft toggles (in `main.tex`)
- `\showtodotrue` → visible red `[TODO: ...]` notes (skeleton). Flip to `\showtodofalse` for a clean PDF.
- `\showclaimtagsfalse` → claim tags hidden in the PDF; flip to `true` to see `\claimtag{...}` traceability.

## Figures (`figures/`)
- `fig1_taxonomy_boundary.png` — shift taxonomy + abstention boundary (reused, current).
- `fig2_pipeline_gates.png` — pipeline/gates (Route-A **placeholder**; TODO: regenerate a B3-specific pipeline).
- `fig3_dev_vs_confirmatory.png` — Route-A dev vs frozen confirmatory (current).
Regenerate/extend via `csc/tools/make_paper_figures.py`.

## Provenance of the numbers
- Route A: tag `csc-confirmatory-v1` / `dee8958`; `csc/results/confirmatory.json`; `notes/CSC_CONFIRMATORY_RESULT.md`.
- Route B3: tag `csc-b3-confirmatory-v1` / `0595f64`; `csc/results/b3_confirmatory_result.json`;
  `notes/CSC_B3_CONFIRMATORY_RESULT.md` + `notes/CSC_B3_CONFIRMATORY_C6_REDTEAM.md`.

Both tags are **locked**; no reruns. All results are **synthetic**; real EEG is future work requiring a
separate freeze and authorization.
