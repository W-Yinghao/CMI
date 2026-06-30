# CIGL Figure Captions (Phase 4D draft)

> Caption drafts only — **no figure assets generated** (writing only). Each caption states the visual
> elements, the claim it supports, the claim it must **not** be used for, and the source evidence.

## F1 — CIGL pipeline

**Caption.** *CIGL pipeline.* Raw EEG `x[B,C,T]` → DGCNN temporal stem → **static (shared) adjacency** →
Chebyshev graph convolution over electrode nodes → per-electrode node features `node_z` → graph readout
`graph_z` → task logits. Conditional-domain posterior heads `q_g`, `q_n` are fit on detached features
(Step A); the encoder is penalized by the graph/node posterior-KL proxies `R_g`, `R_n` (Step B). The
adjacency is shared across samples, so `edge_logits = None` (no per-sample edge object; no edge term).
- **Visual:** boxes/arrows for stem→adjacency→ChebConv→node_z→graph_z→logits; dashed Step-A probe boxes;
  `λ_g R_g`, `λ_n R_n` penalty arrows; an explicit "edge_logits = None / no edge term" annotation.
- **Supports:** the graph/node-only method on a static-adjacency backbone; where the proxies attach.
- **Does NOT support:** any edge-CMI / dynamic-edge claim; any accuracy/SOTA claim.
- **Source:** `docs/CIGL_32`; `cmi/models/graph_task_backbones.py`; `cmi/train/trainer.py` (graphcmi branch).

## F2 — Leakage reduction vs source-task retention

**Caption.** *Leakage reduction at task retention.* Each point is one LOSO fold; x = graph (or node) KL
reduction (%) of `graph_node_010` vs fold/seed-matched ERM, y = source bAcc drop. Markers distinguish
BNCI2014_001 (folds 1–8) and BNCI2015_001 (12 folds). Reference lines: retention gate (drop = 0.02) and
the ≥30% reduction threshold. Points cluster at large reduction (35–77%) with near-zero drop; the single
BNCI2015_001 fold above the 0.02 line (fold9, +0.024) is highlighted.
- **Visual:** scatter, two marker styles, two reference lines, fold9 annotated.
- **Supports:** "partial reduction while meeting the retention gate," and the honest fold9 exception.
- **Does NOT support:** leakage elimination (no point at 100% reduction); accuracy improvement (drops are
  ≈0, not negative-by-design); SOTA.
- **Source:** `STATISTICAL_SUMMARY_DRAFT.md` / CIGL_29 / CIGL_31 per-fold values.

## F3 — Graph/node leakage audit with retrained permutation null

**Caption.** *Source-only leakage audit.* For a trained backbone, fresh held-out conditional-domain probes
`q(D | Z, Y)` are fit on frozen `graph_z`/`node_z`; the posterior-KL proxy is compared to a **within-label,
retrained permutation null** (domain labels permuted within label on the probe-training split only). Bars
show observed vs null KL for graph and node (clearing the null in 3/3 seeds on the DGCNN backbone); an
inset shows the per-electrode node-leakage map (stable across seeds, corr ≈ 0.945).
- **Visual:** grouped bars (observed vs null, graph & node) with significance; electrode node-map inset.
- **Supports:** that the audited leakage is significant and spatially stable; the null is a proper control.
- **Does NOT support:** an unbiased CMI value (this is a proxy); edge leakage (skipped, static adjacency).
- **Source:** `docs/CIGL_25` (3A-H); `cmi/eval/graph_leakage.py::audit_graph_node_objects`.

## F4 — Negative-results decision flow

**Caption.** *Why a static-adjacency DGCNN backbone (negative results).* Flow: original graph backbone
near-chance under source-only (3A-R) → known-good decoders confirm the protocol is learnable (3A-S) →
graph-backbone redesign: only the static-adjacency DGCNN learns the task while dynamic-edge designs overfit
(3A-G) → DGCNN graph/node leakage audit (3A-H) → fixed graph/node regularizer (3A-I/J/K). Each node carries
the gate decision.
- **Visual:** top-to-bottom decision flow with pass/fail badges; the "dynamic-edge overfit" branch marked
  "consistent with fingerprint risk, not causal."
- **Supports:** the methodology and scope (why graph/node-only on this backbone); negatives as evidence.
- **Does NOT support:** that `A(x)` is causally the leakage source; any edge-CMI/dynamic-edge claim.
- **Source:** `docs/CIGL_18/21/23/25`; `docs/CIGL_33`.
