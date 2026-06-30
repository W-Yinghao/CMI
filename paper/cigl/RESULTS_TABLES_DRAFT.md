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

## T3–T4 — Cross-dataset confirmation of fixed `graph_node_010` (CIGL_29 / CIGL_31)

Chance = 0.25 (BNCI2014_001, 4-class) and 0.50 (BNCI2015_001, binary). bAcc = balanced accuracy. KL
reduction = `(ERM − reg)/ERM` of the posterior-KL leakage proxy, per fold/seed-matched. 95% CIs are
fold-level bootstrap (seed 0, 10 000 resamples; see `STATISTICAL_SUMMARY_DRAFT.md`). "retain" = source drop
≤ 0.02 absolute. Edge: skipped (static adjacency; no per-sample edge object). "meets retention gate" is
gate-based, not zero-cost (one BNCI2015_001 fold misses the per-fold threshold; dataset-level gate passes).

| field | **T3 BNCI2014_001** | **T4 BNCI2015_001** |
|---|---|---|
| fold group | primary folds 1–8 (fold-0 = dev, excluded) | all LOSO folds |
| n_folds | 8 | 12 |
| ERM source bAcc (mean [range]) | 0.484 [0.457, 0.508] | 0.706 [0.682, 0.734] |
| reg source bAcc (mean, 95% CI) | 0.488 [0.471, 0.505] | 0.700 [0.693, 0.707] |
| source retention (drop ≤0.02) | **8/8** | **11/12** (fold9 +0.024) |
| graph KL reduction (mean, 95% CI; range) | 44.0% [39.5, 49.0]; 35–58% | 66.2% [60.6, 71.0]; 43–77% |
| node KL reduction (mean, 95% CI; range) | 36.9% [33.6, 40.3]; 31–45% | 51.9% [47.8, 55.6]; 37–61% |
| ERM leakage clears null | 8/8 | 12/12 |
| reg reduces ≥30% (≥2/3 seeds) | 8/8 | 12/12 |
| target guardrail (drop ≤0.05) | 8/8 | 12/12 |
| reg leakage still clears null | **every fold (partial, not elimination)** | **every fold (partial, not elimination)** |
| edge status | skipped (edge_logits=None) | skipped (edge_logits=None) |
| decision | **A** (primary folds 1–8) | **A** (`confirmed_with_target_guardrail=true`) |

## T5 — Negative-results summary

| phase | finding | rules out | shapes the method |
|---|---|---|---|
| 3A-R (CIGL_18) | GraphCMINet ERM near-chance (src ≈ 0.33 < 0.45); 6 repairs fail | "any graph backbone hosts the tradeoff" | need a task-capable backbone |
| 3A-S (CIGL_21) | EEGNet/Shallow/Deep/DGCNN clear ≥0.45; GraphCMINet 0.334 | "the protocol/data are unlearnable" | GraphCMINet is the bottleneck |
| 3A-G (CIGL_23) | only static DGCNN adapter passes; dynamic-edge stems overfit | "dynamic-edge / edge-CMI works here" | graph/node-only on static DGCNN |

> Caveat (apply to T3/T4): the *regularized* leakage still clears the null in every confirmation fold —
> these are **partial** reductions, not elimination. Exact per-fold mean ± CI: see
> `STATISTICAL_SUMMARY_DRAFT.md` (computed from JSON) — `TODO: finalize CI for the paper`.
