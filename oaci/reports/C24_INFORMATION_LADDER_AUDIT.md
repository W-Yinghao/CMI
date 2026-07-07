# C24 — Calibration Information Ladder / Identifiability Boundary Audit (frozen C19 `664007686afb520f`)

> C23 closed the source-only target-free gauge (`G5_offset_source_unobservable`). C24 asks what information, if any, breaks the per-target score-offset non-identifiability. Read-only rungs (R0/R1/R2/R5/R6 + witnesses) below; R3/R4 (target-UNLABELED) require a no-retraining target-audit re-inference behind a P0 replay-identity gate and are NOT proxied, NOT finalized here. DIAGNOSTIC-ONLY; not a selector, not DG success.

- **STAGE: read-only (R3/R4 pending P0-gated re-inference)**
- established read-only: **I1_source_only_nonidentifiable, I6_oracle_only_boundary, I4_few_labels_recover_offset**  ·  unresolved: I2_unlabeled_target_recovers_offset, I3_unlabeled_target_insufficient
- primary (provisional): **`I6_oracle_only_boundary`** — target-centered/rank grouping recovers pooled transport (via 0-label TRANSDUCTIVE target grouping == the target-centered oracle; competence labels only refine it -> the missing ingredient is target GROUPING, not source observability and not primarily labels); source-only summaries do not.

## HARD GATE — target-identity leakage (reported FIRST)

- 9-way target-id accuracy from raw source features **+0.541** vs chance **+0.111** → identity-separable: **True** (gates any positive R3/R4 recovery claim as I7 unless it generalizes LOTO).

## Information ladder (pooled AUC; oracle rungs use target grouping = NON-deployable)

| rung | information | pooled AUC | gap closed | status |
|---|---|---:|---:|:--|
| R0 raw_pooled | none | +0.543 | +0.000 | computed |
| R1 source_only_gauge | source_only | +0.468 | -0.825 | computed |
| R2 source_risk_static_gauge | source_only | +0.521 | -0.245 | computed |
| R3 target_unlabeled_gauge | target_unlabeled_transductive | n/a | n/a | REQUIRES_REINFERENCE |
| R4 source_plus_target_unlabeled_gauge | target_unlabeled_transductive | n/a | n/a | REQUIRES_REINFERENCE |
| R5 few_label_target_calibration | target_labeled_supervised_calibration | +0.671 | +1.415 | computed |
| R6 target_centered_rank_oracle | target_identity_oracle | +0.634 | +1.000 | computed |

- R0 raw pooled **+0.543** → R6 target-centered oracle **+0.634** (oracle gap **+0.091**); within-target ceiling +0.659.
- R1 source-only gauge gap closed **-0.825** (C23: source-only fails/hurts); R2 risk-family gap **-0.245**.

## C24-A — source-only non-identifiability witnesses

- Mantel corr(source-dist, offset-dist): all-pairs **+0.343** (p +0.005) → **CROSS-TARGET +0.212** (p +0.020); within-target block confound: **True**
- source predicts offset (cross-target ≥0.3): **False**; 9 near-source/divergent-offset collisions → source non-identifying: **True**  · CROSS-TARGET source distance does NOT predict offset distance (Mantel 0.212, p 0.020) — all-pairs 0.343 was a within-target block artifact; 9 near-source/divergent-offset collisions -> source is non-identifying for the offset

## R5 — few-label target calibration diagnostic (NON-DG supervised)

| k labels/class | pooled AUC | gap closed |
|---:|---:|---:|
| 0 | +0.634 | +1.000 |
| 1 | +0.630 | +0.954 |
| 2 | +0.654 | +1.227 |
| 4 | +0.667 | +1.365 |
| 8 | +0.671 | +1.415 |

- k=0 is the LABEL-FREE transductive target-mean centering (== oracle): gap closed **+1.000** → offset recovered by target GROUPING at 0 labels; grouping==oracle: **True**.
- competence labels only REFINE beyond grouping (label gain over grouping +0.415); few-labels (≤4/class) recover: **True** (max gap +1.415).

## R3/R4 — target-unlabeled gauge (REQUIRES RE-INFERENCE; not proxied, not finalized)

- status: **REQUIRES_REINFERENCE**. cached target logits are METHOD-FINAL checkpoints (~4 per seed x target x level), NOT the ~60 per-seed x target feasible-OACI CANDIDATE checkpoints the offset is defined over -- using them as R3/R4 would swap the population; REFUSED as science.
- cached method-final target_audit.npz: 216 (wrong population); per-candidate target-unlabeled ready: False.
- next: C24-R3R4-P0 replay-identity smoke gate → full no-retraining target-audit re-inference → real R3/R4.

## Boundary of the claim

> DIAGNOSTIC-ONLY. Oracle rungs use target grouping (non-deployable). R5 is a supervised label-budget diagnostic, not DG. No selector, no selected-checkpoint artifact. C24 is NOT finalized until the P0-gated re-inference supplies R3/R4.