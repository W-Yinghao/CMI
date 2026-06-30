# CIGL_38 — Phase 4D Manuscript v0.2 Review

> Phase 4D summary (writing/analysis only; no GPU, no training, no experiments, no CITA/DualPC/Tri-CMI
> changes). Artifacts under `paper/cigl/`.

## Wording fixes

- Replaced **"without harming source-task performance"** (Abstract + Conclusion of `MANUSCRIPT_DRAFT.md`)
  with **"while meeting the pre-specified source-task retention criteria."**
- `CLAIMS_AUDIT.md` "Task retained" allowed wording → **"meets the pre-specified source-task retention
  gate; one BNCI2015_001 fold misses the per-fold retention threshold but the dataset-level gate passes"**
  (status READY). The guard test now also rejects "without harming source-task performance".
- Hyphenation artifacts (`conditional- information`, `domain- adversarial`, `dynamic- edge`): **none
  present** (checked).

## Table readiness

`RESULTS_TABLES_DRAFT.md` now has paper-facing **Markdown pipe tables** T1–T5. T3/T4 are a combined
cross-dataset table with the required fields: fold group, n_folds, ERM/reg source bAcc (mean [range] / mean
[95% CI]), source-retention count, graph/node KL reduction (mean, 95% CI, range), ERM-leakage / reg-reduces
/ target-guardrail counts, "reg leakage still clears null", edge status, decision.

## CI status

Computed from **existing** per-fold summary JSON (no new runs); **fold-level bootstrap, 10 000 resamples,
percentile 2.5/97.5, seed 0** (`numpy.random.default_rng(0)`):

- BNCI2014_001 (primary folds 1–8): reg source 0.488 [0.471, 0.505]; source drop −0.002 [−0.007, 0.003];
  graph reduction 44.0% [39.5, 49.0]; node reduction 36.9% [33.6, 40.3].
- BNCI2015_001 (12 folds): reg source 0.700 [0.693, 0.707]; source drop +0.007 [+0.001, +0.012]; graph
  reduction 66.2% [60.6, 71.0]; node reduction 51.9% [47.8, 55.6].

Method/seed recorded in `STATISTICAL_SUMMARY_DRAFT.md`. A per-seed (folds×seeds) CI is marked
`TODO: finalize`. The stdlib collector regenerates the per-fold **tables** (no numpy/CI in it, by design);
CIs are the documented numpy fold-bootstrap.

## Citation status

`REFERENCES_VERIFIED.md`: 8 method/benchmark refs verified at title/venue/year (MOABB, EEGNet,
Schirrmeister, DGCNN, RGNN, LGGNet, Li 2018 conditional-invariant, CCMI); exact vol/DOI still `TODO: verify`.
**Dataset primary citations (BNCI2014_001 2a, BNCI2015_001 Faller 2012) remain TODO.** Total open citation
TODOs ≈ 10 fields. No BibTeX finalized; nothing fabricated.

## Figure captions

`FIGURE_CAPTIONS_DRAFT.md` drafts F1 (pipeline), F2 (reduction vs retention), F3 (audit + permutation null),
F4 (negative-results flow); each states visual elements, claim supported, claim NOT supported, and source.
No figure assets generated.

## Ablation memo recommendation

`ABLATION_DECISION_MEMO.md`: graph-only vs node-only at λ=0.010 is **not paper-critical** under the
paired-penalty framing; existing Phase 3A-I fold-0 pilot data **partially** speak to it (small single-term
λ ≈ no reduction; paired 0.010 reduces) but do not cleanly isolate terms at 0.010. **Recommendation: do not
run now**; revisit a single minimal reviewer-gated `graph_010` vs `node_010` run only if review deems the
per-term mechanism a fatal gap.

## Remaining manuscript blockers (all writing/analysis — no GPU)

1. Resolve remaining citation `TODO: verify` (esp. dataset primaries) before submission.
2. Finalize per-seed (folds×seeds) CIs for camera-ready (currently fold-level CIs done).
3. Generate the F1–F4 figure assets (captions ready).
4. Human writing pass for prose flow (v0.3).

## Confirmations

- No GPU / no training / no new dataset / no ablation / no λ-grid in Phase 4D.
- No edge-CMI, no SOTA framing, no CITA/DualPC/Tri-CMI changes, no generated tables committed (gitignored).
- Validation: py_compile OK; collector + manuscript-claims tests pass; collect dry-run OK; wording-check
  block passes.

## Recommendation (pending reviewer)

**A — manuscript v0.2 ready for a human writing pass (v0.3).** No experiment is needed; the remaining items
are citation/CI/figure finalization and prose polish. Authorize the `graph_010` vs `node_010` ablation only
if a human review finds the per-term mechanism essential.
