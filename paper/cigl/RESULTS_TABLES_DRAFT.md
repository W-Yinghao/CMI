# CIGL Results Tables (Phase 4C draft)

> Paper-facing table drafts from **existing** evidence only (CIGL_25/29/31 + tracked summary JSON via
> `scripts/collect_cigl_evidence_tables.py`). No new runs. Numbers are aggregates computed from the summary
> JSON; where a value is not yet computed it says `TODO: compute from generated JSON` (no invented numbers).

## T1 — Method / config / protocol

| field | value |
|---|---|
| backbone | DGCNN static-(shared-)adjacency adapter (`forward_graph → graph_z, node_z, edge_logits=None`) |
| loss | `L_CE + λ_g R_g(Z_g;D\|Y) + λ_n R_n(Z_v;D\|Y)`, **λ_g=λ_n=0.010, λ_edge=0** |
| regularizer | posterior-KL plug-in proxy (NOT unbiased CMI); Step-A posteriors on detached features, Step-B encoder penalty |
| setting | strict source-only DG; target labels evaluation-only |
| datasets | BNCI2014_001 (4-class, chance 0.25); BNCI2015_001 (binary right_hand/feet, chance 0.50) |
| preprocessing | MotorImagery; resample 128 Hz; window 0.5–3.5 s; per-trial z-score |
| seeds / n_perm / gate | 0,1,2 / 50 / α=0.05 |

## T2 — DGCNN leakage audit (CIGL_25, BNCI2014_001 fold-0)

| object | kl_mean | perm_mean | p | clears (seeds) |
|---|---|---|---|---|
| graph `Z_g` | ≈ 1.26 (≈ 8× perm) | ≈ 0.16 | 0.020 | 3/3 |
| node `Z_v` | ≈ 0.52 (≈ 15× perm) | ≈ 0.034 | 0.020 | 3/3 |

Node-leakage map stable across seeds (mean pairwise corr ≈ 0.945 ≫ null q95 ≈ 0.20). Edge: skipped
(static adjacency; `edge_logits=None`).

## T3 — BNCI2014_001 confirmation (CIGL_29; primary folds 1–8; fold-0 = dev, excluded)

| metric | value |
|---|---|
| ERM source bAcc (mean) | ≈ 0.48 (chance 0.25) |
| reg source bAcc (mean) | ≈ 0.49 |
| source drop ≤ 0.02 | **8/8** primary folds (mean drop ≈ −0.00) |
| graph KL reduction | **35–58%** (mean ≈ 44%) |
| node KL reduction | **31–45%** (mean ≈ 37%) |
| criteria (erm adequacy / leakage / reduces / source-retained) | **8/8 each** |
| edge | skipped (no per-sample edge object) |
| verdict | Decision **A** (primary = folds 1–8) |

## T4 — BNCI2015_001 confirmation (CIGL_31; all 12 LOSO folds)

| metric | value |
|---|---|
| ERM source bAcc (mean) | ≈ 0.71 (chance 0.50) |
| reg source bAcc (mean) | ≈ 0.70 |
| source drop ≤ 0.02 | **11/12** folds (fold9 drop +0.024 — the only retention miss) |
| graph KL reduction | **43–77%** (mean ≈ 66%) |
| node KL reduction | **37–61%** (mean ≈ 52%) |
| erm adequacy / leakage / reduces / target-guardrail | **12/12 each**; source-retained **11/12** |
| edge | skipped (no per-sample edge object) |
| verdict | `confirmed_with_target_guardrail = true` → Decision **A** |

## T5 — Negative-results summary

| phase | finding | rules out | shapes the method |
|---|---|---|---|
| 3A-R (CIGL_18) | GraphCMINet ERM near-chance (src ≈ 0.33 < 0.45); 6 repairs fail | "any graph backbone hosts the tradeoff" | need a task-capable backbone |
| 3A-S (CIGL_21) | EEGNet/Shallow/Deep/DGCNN clear ≥0.45; GraphCMINet 0.334 | "the protocol/data are unlearnable" | GraphCMINet is the bottleneck |
| 3A-G (CIGL_23) | only static DGCNN adapter passes; dynamic-edge stems overfit | "dynamic-edge / edge-CMI works here" | graph/node-only on static DGCNN |

> Caveat (apply to T3/T4): the *regularized* leakage still clears the null in every confirmation fold —
> these are **partial** reductions, not elimination. Exact per-fold mean ± CI: see
> `STATISTICAL_SUMMARY_DRAFT.md` (computed from JSON) — `TODO: finalize CI for the paper`.
