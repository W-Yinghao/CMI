# CIGL Ablation Decision Memo (Phase 4D)

> Decision memo only — **no experiment run or recommended-by-default.** Question: is the **graph-only vs
> node-only at λ=0.010** ablation paper-critical, and can existing data answer it? Default position
> (reviewer's): *do not run unless the manuscript cannot stand without it.*

## Q1 — Is graph-only vs node-only at λ=0.010 paper-critical?

**No, not for the bounded claim.** The paper's claim is a **paired graph+node** regularizer
(`graph_node_010`) that reduces graph/node leakage at task retention. It does **not** claim to know which
term drives the reduction. Framed as a *paired penalty*, the result stands without decomposing the terms.
The ablation would *strengthen* mechanism understanding but is not load-bearing for the main claim.

## Q2 — Can existing Phase 3A-I pilot data partially answer it?

**Partially, yes — on fold-0 only.** The Phase 3A-I pilot (CIGL_27, BNCI2014_001 fold-0) already ran the
λ-ladder including **`graph_001` (λ_g=0.001, λ_n=0)**, **`node_001` (0, 0.001)**, **`graph_003`**,
**`node_003`**, and the paired **`graph_node_003`/`graph_node_010`**. From that single dev fold: the
small graph-only / node-only settings (≤0.003) produced **near-zero reduction**, and only the paired
`graph_node_010` achieved the ≥30% reduction. So existing data suggest, **on fold-0**, that neither term
alone at the small λ reduced leakage and the effect appeared at the paired 0.010 setting — but this is
**one dev fold** and does **not** include graph-only/node-only **at λ=0.010** (only the paired config was
run at 0.010). It is suggestive, not a clean isolation.

## Q3 — Would the paper collapse without this ablation?

**No.** With the paired-penalty framing and the two-dataset confirmation (CIGL_29/31), the contribution
(audit + task-preserving partial reduction) is intact. A reviewer might *request* the decomposition; we can
pre-empt by (a) framing as a paired penalty and (b) citing the fold-0 pilot observation above as
preliminary. The paper does not assert a per-term mechanism, so omitting the ablation is not a
contradiction — only a scoped-down mechanistic claim.

## Q4 — Minimal GPU run if later authorized

Two fixed configs added to the existing confirmation runner — **`graph_010` (λ_g=0.010, λ_n=0)** and
**`node_010` (λ_g=0, λ_n=0.010)** — vs the existing `erm_fixed` and `graph_node_010`, on **BNCI2014_001
folds 1–8** (the primary set; optionally + BNCI2015_001), seeds 0–2, epochs 80, n_perm 50, source-only,
edge skipped. No λ grid (these are the two single-term points at the *already-fixed* 0.010). Reuses the
exact runner/firewall; ≈ the cost of one more confirmation pass. **Reviewer-gated; not requested now.**

## Q5 — What claim would it support?

A **mechanism** statement: whether the graph term, the node term, or only their combination drives the
reduction at λ=0.010 (e.g., "the node term is necessary/sufficient" or "the two are complementary"). It
would let the paper say *which* object's regularization carries the effect.

## Q6 — What would remain unsupported even with it?

Everything currently out of scope: edge-CMI / dynamic-edge, cross-architecture generality, beyond-MI,
SOTA, leakage elimination, unbiased CMI, λ-robustness. The ablation isolates terms at one λ on this
backbone; it does **not** extend the scope.

## Recommendation

**Do NOT run now.** Frame CIGL as a **paired graph/node penalty**; report the fold-0 pilot observation
(graph-only/node-only at small λ ≈ no reduction; paired 0.010 reduces) as preliminary and clearly limited.
Revisit a **single, minimal, reviewer-gated** `graph_010` vs `node_010` run only if a manuscript review
deems the per-term mechanism a fatal interpretability gap.
