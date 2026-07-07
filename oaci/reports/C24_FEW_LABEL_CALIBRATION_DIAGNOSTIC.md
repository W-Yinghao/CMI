# C24 R5 — Few-label target calibration diagnostic

> NON-DG supervised target-calibration diagnostic; reveals competence labels for a few of the held-out target's own candidates to estimate its offset. Not deployment, not a selector. NB: k=0 is the LABEL-FREE transductive target-mean centering (== the target-centered oracle); positive labels only refine it. The offset is a target-GROUPING quantity, recoverable transductively with 0 labels once target candidates can be pooled.

- raw pooled +0.543; target-centered oracle +0.634
- few labels (≤4/class) recover: **True**; max gap closed +1.415

| k/class | pooled AUC | gap closed | targets w/ both classes |
|---:|---:|---:|---:|
| 0 | +0.634 | +1.000 | 9/9 |
| 1 | +0.630 | +0.954 | 9/9 |
| 2 | +0.654 | +1.227 | 9/9 |
| 4 | +0.667 | +1.365 | 9/9 |
| 8 | +0.671 | +1.415 | 9/9 |

> Interpretation: if a small labeled budget sharply recovers the offset, the missing quantity is target-specific scalar calibration information. This is a supervised label-budget diagnostic, NOT domain generalization and NOT a selector.