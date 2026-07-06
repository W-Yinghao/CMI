# CIGL_68 — Direct-reliance CMI seed0 gate readout

```
SEED0 SCREENING ONLY — NOT method-level judgment (that needs seeds 0/1/2). 42 runs, replay_ok 42/42, 0 NaN,
random_subspace control ~0, projector firewall source-train-only. Comparators FROZEN (CIGL_65 seed0 + CIGL_67
seed0). PRIMARY metric = R3 task_drop k2 (the counterfactual reliance every proxy failed to move).
CIGL_67 precedent burned in: FCIGL's seed0 R3 signal did NOT survive seeds 0/1/2 — do not over-read this.
```

## Results (fold-mean, seed0)

**BNCI2015_001 (clean readout, target ≈ 0.59):**
| method | target | graph_kl | **R3 k2** | align_k2 | class |
|---|---|---|---|---|---|
| ERM (frozen) | 0.592 | 1.115 | 0.020 | 0.046 | — |
| CIGL (frozen) | 0.586 | 0.314 | **0.082** | 0.529 | — |
| FCIGL-align η0.01 (frozen) | 0.582 | 0.413 | 0.063 | 0.342 | (CIGL_67: alignment↓, R3 NS across seeds) |
| **dcigl_consistency β0.1** | 0.586 | 0.479 | **0.050** | 0.662 | **STRONG_RELIANCE** |
| dcigl_consistency β0.5 | 0.584 | 0.459 | 0.065 | 0.623 | functional |

**BNCI2014_001 (near-chance, target ≈ 0.33 — weak reliance readout):**
| method | target | R3 k2 | class |
|---|---|---|---|
| CIGL (frozen) | 0.334 | 0.023 | — |
| dcigl β0.1 | 0.341 | 0.020 | fail (R3 barely moves) |
| dcigl β0.5 | 0.339 | 0.026 | fail |

## Read (honest, seed0)
1. **On 2015, the DIRECT reliance objective reduces R3 where both proxies failed.** dcigl β0.1: R3 task_drop
   0.082 → **0.050 (−39%)**, **below FCIGL-align (0.063)** and far below CIGL (0.082); **target retained** (0.586 =
   CIGL); leakage below ERM (0.479 << 1.115); random control ≈ 0. This is the first seed0 signal that optimizing
   the counterfactual reliance itself moves R3.
2. **Mechanistic contrast — R3 and alignment are dissociable.** dcigl reduces R3 **without** reducing alignment
   (align 0.529 → 0.662, if anything higher). FCIGL did the opposite (alignment 0.529 → 0.342, R3 unchanged). So
   the alignment scalar was only a diagnostic correlate (CIGL_67), and the direct R3 objective is what moves
   reliance — exactly the hypothesis this probe set out to test.
3. **β0.1 > β0.5.** The smaller consistency weight gives the larger R3 reduction (0.050 vs 0.065) at equal task;
   β0.5 over-regularizes. Best variant = **β0.1**.
4. **2a (near-chance)** cannot carry the reliance claim (R3 barely moves for anyone). Directionally neutral.

## CRITICAL caveat
**This is seed0 only.** FCIGL's seed0 R3 signal on 2015 (0.082→0.063) looked good and **did not replicate across
seeds 0/1/2**. dcigl's seed0 signal is *stronger* (0.050, below FCIGL) but is subject to the same risk. **No
method-level claim.** Per the project rule, method-level judgment requires full-LOSO × seeds 0/1/2.

## Recommendation (PM decides)
**Expand only the best β (β0.1) to seeds 1/2, both datasets** (per the PM expand rule: 2015 shows R3 ↓ clearly,
below FCIGL, task retained; 2a target not hurt). This directly tests whether the R3 reduction is **stable** — the
exact thing FCIGL failed. Do **not** expand β0.5 (weaker, over-regularizes). If seeds 1/2 confirm, this is the
first stable CMI-driven reliance reduction; if it collapses like FCIGL, **stop the CMI method search** and freeze
the three-level measurement→reliance gap as the scientific synthesis.

## Artifacts (`results/cigl_direct/seed0/`)
`dcigl_seed0_metrics.csv` (42), `dcigl_seed0_vs_frozen.csv`. Analysis: `scripts/analyze_dcigl_seed0.py`.
```
```
