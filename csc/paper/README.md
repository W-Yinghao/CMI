# CSC manuscript (`csc/paper/`)

**Title.** *When Can Concept Shift Be Certified? From Unlabeled Abstention to Paired Minimal-Label
Certificates.* (A case study in EEG domain shift.)

**Status.** Skeleton. Claims, scope, numbers, and tables are locked to committed artifacts (see
`claim_evidence_table.md`); prose is stubbed with visible `[TODO: ...]` markers. This is not a finished
paper.

## The story (5-part main line)
1. **Theory** — Z-only non-identifiability of concept shift; abstention is necessary (`sections/theory.tex`, Prop. 1).
2. **Route A audit** — source-anchored Z-only certificate; frozen confirmatory FAIL on unseen clusters (`sections/routeA.tex`).
3. **Route B3 method** — paired within-subject target-internal reference + minimal labels (`sections/routeB3_method.tex`).
4. **Route B3 confirmatory** — frozen unseen synthetic PASS (C1–C6) within a scoped envelope (`sections/routeB3_confirmatory.tex`).
5. **Boundary** — synthetic-only, envelope limits, pure-conditional tail, real-EEG as future work (`sections/discussion.tex`).

**Target venue: AAAI-27 main technical track.** Strategy: quality-first master draft now; page compression
(7-page main content) is a later phase, not now.

## Build

```bash
bash csc/paper/compile.sh        # neutral build  -> main.pdf  (self-contained, base LaTeX, single column)
bash csc/paper/compile_aaai.sh   # AAAI build     -> aaai_main.pdf + page count (two-column)
bash csc/paper/check_claims.sh   # forbidden-wording + required-scope + compile gate
```

### Two-build convention
Mirrors `tos_cmi/paper`: `main.tex` (neutral, single-column, for reading/review) and `aaai_main.tex`
(AAAI-oriented, two-column) share `sections/` verbatim.

**AAAI style file (Phase 2 swap).** `aaai_main.tex` is currently a two-column `article` *approximation* so
the paper compiles and page count is trackable now. The official AAAI-27 kit (`aaai27.sty` + `aaai27.bst`)
is copyrighted and is **not** in this workspace (the sibling `AAAI_2026/nab_overleaf/nab_aaai.tex` uses the
same article stand-in). At submission: drop the kit files here and replace the `aaai_main.tex` preamble per
the kit. Phase 1 (now): compiles, anonymous, claims unchanged, may exceed 7 pages. Phase 2 (later): official
style + compression (proofs / large tables / C6 detail move to appendix/supplement).

## Draft toggles (in `main.tex`)
- `\showtodotrue` → visible red `[TODO: ...]` notes (skeleton). Flip to `\showtodofalse` for a clean PDF.
- `\showclaimtagsfalse` → claim tags hidden in the PDF; flip to `true` to see `\claimtag{...}` traceability.

## Figures
Planned 3 main figures for AAAI (Route-A pipeline demoted to appendix):
- **Fig 1** `figures/fig1_taxonomy_boundary.png` — shift taxonomy + non-identifiability/abstention boundary.
- **Fig 2** — Route-B3 paired minimal-label **pipeline**, now a native **TikZ** figure inline in
  `sections/routeB3_method.tex` (no external asset). The old `figures/fig2_pipeline_gates.png` (Route-A
  pipeline) is retained for possible appendix use.
- **Fig 3** `figures/fig3_dev_vs_confirmatory.png` — frozen evidence summary (Route-A dev→confirmatory
  collapse; TODO: add a Route-B3 C1–C6 PASS panel alongside).
Raster figures regenerate/extend via `csc/tools/make_paper_figures.py`.

## Provenance of the numbers
- Route A: tag `csc-confirmatory-v1` / `dee8958`; `csc/results/confirmatory.json`; `notes/CSC_CONFIRMATORY_RESULT.md`.
- Route B3: tag `csc-b3-confirmatory-v1` / `0595f64`; `csc/results/b3_confirmatory_result.json`;
  `notes/CSC_B3_CONFIRMATORY_RESULT.md` + `notes/CSC_B3_CONFIRMATORY_C6_REDTEAM.md`.

Both tags are **locked**; no reruns. All results are **synthetic**; real EEG is future work requiring a
separate freeze and authorization.
