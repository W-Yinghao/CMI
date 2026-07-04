# FBCSP-LGG F1a decoder-only + full-LOSO ERM references (seeds 0/1/2)

147/147 jobs rc=0, 0 NaN, all central_strip_v1 (FBLGG/FBCSP). Same runner, full LOSO.

## Full-LOSO ERM references (seeds 0/1/2) — apples-to-apples backbone comparison
| backbone (erm:0) | 2a mean | 2a worst | 2015 mean | 2015 worst |
|---|---|---|---|---|
| DGCNNGraph    | 0.311 | 0.240 | 0.588 | 0.482 |
| FBLGGGraph    | 0.287 | 0.234 | 0.593 | 0.483 |
| **FBCSPLGGGraph** | **0.349** | 0.236 | **0.608** | 0.460 |

**FBCSP-LGG ERM is the best backbone on BOTH datasets in full-LOSO** (2a +3.8pp vs DGCNN / +6.2pp vs
FBLGG; 2015 +2.0pp / +1.5pp). The earlier "2015 regression vs FBLGG 0.627" was an artifact of comparing
to a 2-FOLD FBLGG; the full-LOSO FBLGG is 0.593, which FBCSP-LGG beats. The spatial branch helps.

## F1a decoder-only I(Y;D|Z) vs FBCSP ERM (seed0, apples-to-apples)
| config (seed0) | 2a mean | 2a worst | 2015 mean | 2015 worst |
|---|---|---|---|---|
| FBCSP ERM | 0.354 | 0.241 | 0.596 | 0.500 |
| FBCSP graphdualpc decoder-only @dec_scale=300 | **0.371** | 0.248 | **0.604** | 0.487 |

Decoder-side CMI improves ERM: **+1.7pp (2a), +0.8pp (2015)** on seed0.

## F1a decoder diagnostics (seed0)
- loss_dec_over_ce = **30.8% (2a), 26.1% (2015)** -> dec_scale=300 is TOO STRONG (target 1-10%; >20% is
  the "too strong" zone). It still helped, but should be recalibrated (~dec_scale 100 -> ~10%).
- dec_gate_active_frac = 1.00; gate g/t/s = 0.29/0.19/0.51 (2a), 0.27/0.16/0.57 (2015) -> spatial still
  dominant, graph ~0.28 (not fully starved); zero_spatial still load-bearing (drop ~0.08-0.09).

## Reading
1. FBCSP-LGG is validated as the strongest backbone (full-LOSO, both datasets) vs DGCNN and FBLGG.
2. Decoder-only CMI is positive on seed0 -> per the F1a gate, run seeds 1/2 to confirm.
3. dec_scale=300 is over-strong (~30% of CE) -> recalibrate toward ~10% (dec_scale ~100) for F1/P6.
4. P6 (fbdualpc spatial-CMI + fusion floor) is the aligned next objective (the load-bearing branch is
   spatial, gate ~0.51-0.57), committed on project/fbcsp-lgg-spatial-cmi-fusion (a66fad5), CPU-tested.
