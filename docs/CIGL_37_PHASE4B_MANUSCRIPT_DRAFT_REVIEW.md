# CIGL_37 — Phase 4B Manuscript Draft Review

> Summary of the Phase 4B manuscript-drafting step (writing only; no GPU, no training, no new experiments,
> no CITA/DualPC/Tri-CMI changes). The reviewable artifacts live under `paper/cigl/`.

## What was drafted

- `paper/cigl/README.md` — draft workspace rules (no PDFs, no experiments, claims trace to CIGL_32–36 / 29 / 31).
- `paper/cigl/MANUSCRIPT_DRAFT.md` — a first full skeleton with paragraph-level prose: Title, Abstract, and
  §1 Introduction · §2 Related Work · §3 Method · §4 Experimental Protocol · §5 Results · §6 Analysis and
  Negative Results · §7 Limitations · §8 Conclusion. Table/figure placeholders `[TABLE: …]`/`[FIGURE: …]`.
- `paper/cigl/CLAIMS_AUDIT.md` — claim → evidence → allowed/forbidden wording → status (11 claims).
- `paper/cigl/TABLES_AND_FIGURES_PLAN.md` — T1–T5, F1–F4 with source, claim supported, claim NOT supported,
  status.
- `paper/cigl/RELATED_WORK_MATRIX.md` — 9 positioning rows; unverified references marked
  `TODO: verify citation` (no fabricated bibliography).
- `paper/cigl/OPEN_PAPER_BLOCKERS.md` — candidate gaps classified (must-fix / optional-ablation /
  future-work / do-not-do).
- `tests/test_cigl_manuscript_claims.py` — guards forbidden affirmative phrasings and required disclaimers.

## Claim boundaries (held in the draft)

The manuscript states the bounded claim and **only** the bounded claim: significant graph/node
label-conditional domain leakage on a task-capable DGCNN static-adjacency backbone, **partially and
reproducibly reduced** by a fixed graph/node posterior-KL regularizer (`λ_g=λ_n=0.010`, no edge term)
**without harming source-task performance**, on two MI datasets, source-only. It explicitly **disclaims**:
SOTA, leakage elimination, unbiased CMI, edge-CMI / dynamic-edge, cross-architecture, beyond-MI, and
λ-robustness. Negative results (GraphCMINet near-chance; dynamic-edge overfitting) are in a **mandatory §6**,
framed as method-shaping evidence. Target labels are **evaluation-only** throughout.

## Unresolved manuscript blockers (all writing/analysis — no GPU)

1. **MUST-FIX — citation verification:** replace every `TODO: verify citation`/`TODO: verify source` with a
   checked reference before submission (no fabricated bibliography).
2. **MUST-FIX — baseline-table clarity:** state ERM ≈0.46 (2a) / ≈0.70 (2015), chance 0.25/0.50, and the
   precise meaning of "task retained" so no accuracy/SOTA claim is implied.
3. **MUST-FIX — aggregate fold statistics:** mean ± CI / counts for reduction & retention, computable from
   the existing summary JSON (no new runs).

## Is any experiment recommended now?

**No.** Per the reviewer's "write first, experiment only if the manuscript cannot stand without it," the
draft stands on the two-dataset confirmation already in hand. The optional ablations (graph-only vs
node-only at λ=0.010; marginal vs conditional) may be **partly answerable from existing Phase 3A-I pilot
data** (its λ-ladder already includes graph-only/node-only at small λ); a new GPU run should be requested
only if the draft demonstrably needs it. A third dataset and a constrained dynamic-edge backbone are
**future work**, not preconditions.

## Confirmations

- **No GPU / no EEG training / no new dataset / no ablation / no λ-grid** in Phase 4B.
- **No** edge-CMI, **no** SOTA framing, **no** CITA/DualPC/Tri-CMI code changes.
- **No** generated PDFs; **no** fabricated citations (all unverified refs marked TODO).

## Recommendation (pending reviewer)

**A — proceed to full manuscript polish** (resolve the three MUST-FIX writing items, generate T2–T4 tables
via `scripts/collect_cigl_evidence_tables.py`, draft F1–F4). Do **not** open a new GPU line unless the
polish surfaces a genuinely paper-critical gap; if it does, request **one** minimal, reviewer-gated
ablation (most likely graph-only vs node-only, possibly satisfiable from existing pilot data).
