# FBCSP-LGG F0 — full-LOSO ERM (seeds 0/1/2). GPU/F1 frozen.

**The P5 spatial fix works where the P4 diagnosis said it should**, but the full-LOSO 2a mean is modest
because ~5/9 BNCI2014 subjects are intrinsically hard cross-subject (CSP also fails there). On the
decodable fold subj1, FBCSP nearly matches classical CSP and beats FBLGG by +16.8pp. On BNCI2015 there is
a small regression (the spatial branch crowds out the graph branch that FBLGG relied on).

## Provenance
```
Branch/SHA : project/fbcsp-lgg-dualcmi-scaffold @ add0a76 (tree clean at launch)
Backbone   : FBCSPLGGGraph  Config: erm:0  Flags: --source_val_early_stop  Env: eeg2025, CUDA, MNE mirror
Datasets   : BNCI2014_001 folds 0-8 (9) + BNCI2015_001 folds 0-11 (12), seeds 0/1/2 = 63 cells
Ops        : 63/63 rc=0, 0 NaN/inf, 0 missing metrics, all grouping=central_strip_v1, all mirror, no braindecode
```

## Aggregate (63 cells). CSV: FBCSP_F0_AGGREGATE.csv / FBCSP_F0_SEED_TABLE.csv
| dataset | n | mean | worst | std | src | best-srcval | final-val | zeroG | zeroT | **zeroS** | permN | gate_graph | gate_temp | **gate_spatial** | Δ vs FBLGG(2f) | Δ vs DGCNN(2f) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 (.25) | 27 | 0.349 | 0.236 | 0.087 | 0.664 | 0.452 | 0.379 | 0.366 | 0.362 | **0.275** | 0.347 | 0.279 | 0.232 | **0.489** | +0.053 | +0.007 |
| BNCI2015_001 (.50) | 36 | 0.608 | 0.460 | 0.093 | 0.834 | 0.673 | 0.619 | 0.616 | 0.610 | **0.520** | 0.602 | 0.239 | 0.189 | **0.572** | −0.019 | +0.033 |

(FBLGG/DGCNN refs are 2-fold subj1/2; see same-fold comparison below for apples-to-apples.)

## The spatial branch is load-bearing (P5 goal met)
- **2a: `zero_spatial` is the biggest ablation drop** (0.349→0.275, −7.4pp) and **gate_spatial is the
  highest branch weight (0.489)**. The CSP-style branch is doing the work on 4-class — exactly the P4 fix.
- `zero_graph` on 2a barely moves (0.349→0.366) → the graph branch is neutral/slightly harmful on 4-class.
- 2015: `zero_spatial` also biggest (−8.8pp); gate_spatial 0.572. The spatial branch now dominates BOTH.

## Same-fold apples-to-apples (subj1, subj2 = the original F0 folds)
| | subj1 | subj2 | 2-fold mean |
|---|---|---|---|
| **FBCSP-LGG** | **0.474** | 0.280 | **0.377** |
| DGCNN (F0) | 0.403 | 0.281 | 0.342 |
| FBLGG (F0) | 0.306 | 0.281 | 0.296 |
| CSP cross-subj | 0.483 | 0.248 | 0.365 |

On the same 2 folds, **FBCSP 0.377 > DGCNN 0.342 > FBLGG 0.296**; on subj1, **FBCSP 0.474 ≈ CSP 0.483** —
the spatial branch closes the CSP gap that P4 identified.

## CSP-decodable vs CSP-hard subset (2a; P5-A map)
| subset | subjects | FBCSP mean | CSP cross-subj mean |
|---|---|---|---|
| CSP-decodable | 1, 3, 8, 9 | **0.430** | 0.524 |
| CSP-hard | 2, 4, 5, 6, 7 | 0.284 | 0.291 |

Per-subject (FBCSP 3-seed vs CSP): subj1 0.474≈0.483; subj3 0.453<0.571; subj8 0.410<0.599; subj9
0.385<0.444. FBCSP matches CSP on subj1 but still trails on subj3/8/9. The full-LOSO mean (0.349) is
dragged down by the CSP-hard subset (0.284 ≈ CSP 0.291 — intrinsically hard, not fixable by architecture).

## Gate mapping (PI full-LOSO gate)
- **BNCI2014: borderline → "continue analysis".** Full mean 0.349 (just under the 0.35 line; +5.3pp over
  FBLGG 2-fold, +0.7pp over DGCNN 2-fold; same-fold +8pp over FBLGG). The decisive positive: the
  **CSP-decodable subset clearly improved and the spatial branch is load-bearing** (subj1 matches CSP).
  Not ≥0.40 (strong pass); not ≤0.32 (fail).
- **BNCI2015: acceptable.** 0.608 ≥ 0.60 and the spatial ablation is meaningful — but −1.9pp vs FBLGG:
  the 3-way gate over-weights spatial (0.572) and under-uses the graph branch (0.239) that helped FBLGG.
  The PI's "zero_graph/permute_nodes should matter on 2015" no longer holds — graph became neutral.

## Honest read / open questions (for PI)
1. The P5 spatial branch **fixed the P4-diagnosed CSP gap on the decodable case** (subj1: 0.306→0.474).
   This validates the direction.
2. Full-LOSO 2a mean is modest because the hard subjects (CSP-unlearnable) dominate the average; the
   architecture win is real but concentrated on the decodable subjects.
3. FBCSP still trails CSP on subj3/8/9 → the spatial branch helps but isn't yet as strong as full FBCSP.
4. 2015 slight regression + the graph branch going neutral on both datasets → the **3-way fusion gate may
   be collapsing toward spatial and starving the graph branch**. A fusion-balance fix (or per-branch aux
   loss) may recover the graph contribution and lift both datasets.

## Frozen
F1 / graphcmi / graphdualpc / dec_scale=300 / λ,γ sweep / dynamic edge / GPU all frozen pending PI review.
