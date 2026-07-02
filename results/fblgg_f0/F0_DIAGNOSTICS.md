# CIGL_48 F0 diagnostics (P4-A/C/D) — BNCI2014 vs BNCI2015 (FBLGGGraph ERM)

Non-GPU analysis of the F0 result (10 JSON + preds sidecars). Localizes the 2a failure.

## P4-D — branch ablation deltas (main − ablation; larger ⇒ more load-bearing)

| dataset | Δ zero_graph | Δ zero_temporal | Δ permute_nodes | reading |
|---|---|---|---|---|
| BNCI2014_001 | +0.023 | +0.012 | +0.020 | all near-zero: no branch carries transferable signal |
| BNCI2015_001 | +0.085 | +0.003 | +0.114 | graph+node load-bearing, temporal not |

## P4-A — early-stopping / source-val behavior

| dataset | best_ep range | best_src_val | target bAcc | final_train | final_val |
|---|---|---|---|---|---|
| BNCI2014_001 | [1,155] | 0.326 | 0.296 | 0.990 | 0.278 |
| BNCI2015_001 | [10,34] | 0.638 | 0.627 | 1.000 | 0.581 |

## P4-C — pooled class-wise confusion (rows=true, cols=pred)

### BNCI2014_001

| true\pred | feet | left_hand | right_hand | tongue | recall |
|---|---|---|---|---|---|
| feet | 164 | 234 | 190 | 276 | 0.190 |
| left_hand | 123 | 311 | 197 | 233 | 0.360 |
| right_hand | 132 | 277 | 247 | 208 | 0.286 |
| tongue | 154 | 234 | 176 | 300 | 0.347 |

pred distribution: feet=573, left_hand=1056, right_hand=810, tongue=1017
one-vs-rest bAcc: feet=0.516, left_hand=0.536, right_hand=0.534, tongue=0.535

### BNCI2015_001

| true\pred | feet | right_hand | recall |
|---|---|---|---|
| feet | 1181 | 319 | 0.787 |
| right_hand | 829 | 671 | 0.447 |

pred distribution: feet=2010, right_hand=990
one-vs-rest bAcc: feet=0.617, right_hand=0.617

## Reading

- **2a failure is diffuse**: all 4 classes weakly separable (OvR ≈ 0.52–0.54), *feet* worst (recall 0.19, mistaken for left_hand/tongue); all 4 classes ARE predicted (no collapse). Not a single-class or specific L/R failure → a representation that lacks transferable 4-class MI structure.
- **2a branch ablations all ~0 (2–3pp)**: neither graph nor temporal branch carries transferable signal (there is little signal to remove). Contrast 2015: graph/node strongly load-bearing (8–11pp).
- **2a early stop**: fits source-train ~0.99 but best-source-val ≈ chance (0.33) at *every* best_ep (1→155) → no epoch generalizes cross-subject. Not an early-stop or grouping bug.
- **P4-F (CSP/LDA classical baseline)**: appended after the CPU job — the decisive architecture-vs-data test.

