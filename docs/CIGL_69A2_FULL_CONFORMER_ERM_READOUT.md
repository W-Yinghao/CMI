# CIGL_69A2 — Full-Conformer ERM readout (seed0 screening + seeds 0/1/2 multi-seed)

```
conformer_full ERM full-LOSO on BNCI2014_001 (9) + BNCI2015_001 (12). seed0 = the PM-approved gate (b, 21 runs,
885688/885689). Seeds 1/2 were an idle-slot ERM fill (885726-729 conformer_full; 885734-737 eegnet+conformer,
the same-protocol comparators) -> the full 3-backbone x 3-seed ERM matrix is complete. Per the standing rule:
seed0 = scientific SCREENING; seeds 0/1/2 = backbone-level JUDGMENT. Isolated dir metacmi_gate_full/ for
conformer_full; frozen CIGL_69A untouched.
```

## 1. Integrity (seed0, 21 folds; seeds 1/2 likewise clean)
21/21 folds finite, no NaN/crashes; feature_z export OK; firewall metadata present (source-only train,
source-val select, target eval-only). Replay mode = **`probe_replay`** on all folds (MLP head → source-fit probe,
`replay_ok=False`), as designed.

## 2. Target performance (seed0)
| dataset | target_bacc | source | train | gap (src−tgt) |
|---|---|---|---|---|
| 2a | 0.428 ± 0.154 | 0.581 | 0.703 | +0.152 |
| 2015 | 0.643 ± 0.097 | 0.789 | 0.908 | +0.146 |

## 3. Comparison — seed0 looked best-of-three, MULTI-SEED OVERTURNS IT
**seed0 target_bacc** (screening only): conformer_full is nominally highest on both (2a 0.428 vs eegnet 0.423 /
mini 0.411; 2015 0.643 vs 0.637 / 0.640) — margins +0.003…+0.017, all within one fold-sd.

**seeds 0/1/2 (backbone-level judgment):**
| | 2a per-seed | 2a pooled | 2015 per-seed | 2015 pooled |
|---|---|---|---|---|
| EEGNetMini | 0.423/0.414/0.421 | **0.419** | 0.637/0.640/0.640 | **0.639** |
| ConformerMini | 0.411/0.405/0.407 | 0.408 | 0.640/0.642/0.635 | 0.639 |
| conformer_full | 0.428/0.367/0.418 | **0.404** | 0.643/0.624/0.620 | **0.629** |

- Δ(conformer_full − EEGNetMini) per seed: **2a +0.005 / −0.047 / −0.003**; **2015 +0.006 / −0.016 / −0.020**.
- conformer_full > EEGNetMini in **1/3 seeds** (only seed0); best-of-three in **1/3 seeds**.
- **The seed0 edge was noise.** Pooled over seeds 0/1/2, conformer_full is *marginally the weakest of the three*
  on BOTH datasets (2a −0.015 vs eegnet; 2015 −0.010). ConformerMini ≥ conformer_full pooled on both.
- conformer_full has **higher seed variance** than the anchor (2a 0.428/0.367/0.418 vs EEGNetMini's stable
  0.423/0.414/0.421).
- **Capacity was NOT the limiter:** the 7.8–10.2× model is not stronger; the Conformer family sits at parity
  with EEGNetMini on these two MI datasets regardless of capacity.
- Regularization note (stated cautiously): conformer_full shows **less source-train saturation** than
  ConformerMini (train 0.70/0.91 vs 0.98/0.995), likely due to dropout-0.5 / MLP regularization, while keeping
  comparable target performance. This is NOT "generalizes better" — the source→target gap is still ~+0.15 and the
  target edge is negative-to-nil.

## 4. Auditability (clean)
- feature_z leakage significant on **all 21 folds** (perm_p<0.05); **leakage-rich** (featKL 2a 1.49 / 2015 1.37,
  ≈ ConformerMini, ≫ EEGNet 1.09/0.68). This is the substrate that makes conformer_full a worthwhile MetaCMI arm.
- **R3 (source-fit probe, k=2), no anomalies:** removal_mode=`probe_replay`, firewall=True on all folds.
  - 2a: task_drop(label-cond) **+0.006 ± 0.011** [src probe 0.933→0.931]; random_subspace control **+0.000 ± 0.000**
    → specificity +0.006. logit margin 2.03 / entropy 0.727 / logit_norm 4.03.
  - 2015: task_drop **+0.008 ± 0.007** [0.956→0.954]; random_subspace **+0.000 ± 0.000** → specificity +0.008.
    margin 4.34 / entropy 0.232 / logit_norm 6.39.
  - Reading: the label-conditional subject subspace carries a **small but SPECIFIC** ERM reliance (random control
    is exactly 0). Logits are non-degenerate (no flat-collapse). CAVEAT: the ERM functional reliance is small, so
    the *headroom* for MetaCMI to reduce it is limited — a null there would be scientifically meaningful, a
    reduction more persuasive on this leakage-rich arm than on a low-leakage one.
- **MLP head exact-replay is FEASIBLE:** in eval mode max|logits − head(feature_z)| = **0.00e+00** (dropout
  identity, ELU deterministic). So Phase-2 ConformerFull R3 CAN be **classifier-level (head-replay)** if we export
  the MLP head params (task_head_kind="mlp" + layer/activation/dropout-eval metadata + replay_max_abs_diff), not
  only representation-level probe. Until that export exists, any ConformerFull R3 claim stays
  **representation-level (probe)**, distinct from EEGNetMini/ConformerMini's exact linear classifier-level replay.

## 5. Verdict — USABLE PASS (qualified), NOT a stronger backbone
By the pre-registered tiers: **not** a strong capacity pass (conformer_full > EEGNetMini in only 1/3 seeds, pooled
below on both); **not** a fail (no collapse, within ~0.01–0.02 of EEGNetMini, artifacts audit/R3-compatible,
R3 clean). ⇒ **USABLE PASS**: a healthy, auditable, leakage-rich, high-capacity *diagnostic* arm — usable to test
whether MetaCMI transfers to a high-capacity transformer — but **parity-to-slightly-below EEGNetMini across seeds,
NOT a proven stronger backbone**, and the seed0 best-of-three is retracted.

### Phase-2 implication (HELD for PM go/no-go)
The R3 is clean and MLP-replay is feasible, so ConformerFull is carry-able into MetaCMI as the high-capacity arm.
BUT the multi-seed correction weakens the "stronger high-capacity arm" rationale: pooled, ConformerMini ≥
ConformerFull and ConformerMini has EXACT linear (classifier-level) replay. So the Phase-2 Conformer arm is a
deliberate choice:
- **ConformerFull** — tests the genuine HIGH-CAPACITY regime + needs the MLP-replay export for classifier-level R3.
- **ConformerMini** — pooled-equal-or-better, exact linear replay already, but lower capacity.
Recommend the PM pick the Conformer arm with this corrected picture. Phase-2 = ERM / MetaCE / MetaCMI-Direct β0.1
on {EEGNetMini + chosen Conformer arm} × {2a, 2015}, seed0, reusing ERM comparators (+~84 runs, no β0.5). No GPU
until PM approves.
```
