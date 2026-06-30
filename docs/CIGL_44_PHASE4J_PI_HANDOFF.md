# CIGL_44 — Phase 4J PI Handoff Packet

> Phase 4J summary (documentation only; no manuscript scientific content changed; no GPU, no training, no
> experiments, no ablation, no template migration, no CITA/DualPC/Tri-CMI changes, no PDF/aux committed,
> no invented citations).

## Purpose

Prepare a focused handoff so the **PI human prose read** (the next mandatory action) is efficient. This
packet adds only reading/triage docs; it does not edit the paper, migrate the TMLR template, or run anything.

## Deliverables

- `paper/cigl_latex/PI_READING_GUIDE.md` — read order, the one-sentence contribution, the exact bounded
  claim, forbidden claims, high-risk paragraphs, the 8 questions for the PI, TMLR-first recommendation.
- `paper/cigl_latex/SECTION_EDIT_QUEUE.md` — per-section purpose / what-to-check / likely weakness /
  expansion-needed / claim-boundary risk; highest-leverage edit order (Related Work first).
- `paper/cigl_latex/TMLR_PREFLIGHT_NOTES.md` — TMLR requirements to confirm (double-blind, TMLR stylefile,
  code/data statement, conflicts, supplementary policy, broader impact), anonymization checklist, and the
  migrate-only-after-prose-read steps. **No migration done.**

## Why no experiment is authorized

The bounded claim (measurement + partial task-preserving control on two MI datasets, graph/node-only, fixed
λ, proxy metric) is fully supported by CIGL_25/29/31, and successive adversarial audits found the paper
stands without any new experiment. The `graph_010` vs `node_010` mechanism ablation remains **optional** and
**unauthorized**; the current risk is prose/story maturity, not missing evidence.

## What the PI should review

The current v0.6 review PDF / LaTeX draft, guided by `PI_READING_GUIDE.md`, answering the 8 questions
(story clarity, intro gap, related-work depth, negatives-as-evidence, method-not-log, results-without-SOTA,
honest-not-self-defeating limitations, TMLR fit).

## What feedback is needed before template migration

The PI's answers to the 8 questions — especially whether Related Work needs expansion and whether the
Introduction makes the measurement→control gap compelling. Template migration (Phase 4K) should start only
after the prose is judged readable, to avoid formatting churn.

## Confirmations

- No GPU / training / experiments / ablation / dataset / λ-grid / edge-CMI / SOTA; no CITA/DualPC/Tri-CMI
  changes; no manuscript scientific content changed (docs-only); no generated PDF/aux committed.
- Validation: handoff docs present (≥20 lines each); `test_cigl_latex_package` + `test_cigl_manuscript_claims`
  pass; no tracked LaTeX artifacts.

## Recommendation

**HOLD for PI prose feedback** before TMLR-template migration (Phase 4K).
