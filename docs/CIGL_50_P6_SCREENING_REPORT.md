# CIGL_50 P6 — FBCSP-LGG-DualCMI full-LOSO seed0 screening report

**Backbone** `FBCSPLGGGraph` · **seed** 0 · **fusion_floor** 0.05 · **early-stop** source_val ·
**worktree** `project/fbcsp-lgg-spatial-cmi-fusion` @ `a66fad5` · 21/21 folds, 0 NaN, all 5 configs/fold.
Deltas are vs the **in-screening `erm:0` control (fusion_floor=0.05)** — the only floor-matched baseline
(see §4). Numbers **independently reproduced to 3 decimals** by a 4-lens adversarial verification
(recompute + skeptic + advocate + data-quality); zero discrepancies.

## 1. BNCI2014_001 (2a, 4-class, 9 folds)   prior ff=0.0 ref = 0.349

| config | mean | worst | Δ floor | src | bstV | finV | zGr | zTe | zSp | pN | gGr | gTe | gSp | gEnt | dec% | rSp | lSp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| erm (floor .05) | 0.361 | 0.243 | — | 0.637 | 0.505 | 0.422 | 0.391 | 0.359 | 0.269 | 0.367 | 0.269 | 0.226 | 0.505 | 0.847 | — | — | — |
| dec-only s50 | 0.366 | 0.238 | +0.004 | 0.626 | 0.519 | 0.415 | 0.396 | 0.366 | 0.268 | 0.375 | 0.342 | 0.209 | 0.449 | 0.849 | 7.0% | 1.609 | 0.0000 |
| dec-only s100 | 0.365 | 0.240 | +0.003 | 0.644 | 0.522 | 0.419 | 0.396 | 0.359 | 0.269 | 0.372 | 0.331 | 0.206 | 0.463 | 0.846 | 12.3% | 1.629 | 0.0000 |
| **spatialCMI .003** | **0.368** | 0.250 | +0.007 | 0.615 | 0.520 | 0.418 | 0.400 | 0.364 | 0.271 | 0.378 | 0.335 | 0.202 | 0.463 | 0.855 | 7.1% | 1.444 | 0.0043 |
| all-CMI g/n/s | 0.358 | 0.238 | **−0.004** | 0.659 | 0.519 | 0.415 | 0.389 | 0.355 | 0.271 | 0.364 | 0.315 | 0.206 | 0.478 | 0.843 | 6.4% | 1.438 | 0.0043 |

## 2. BNCI2015_001 (2015, 2-class, 12 folds)   prior ff=0.0 ref = 0.608

| config | mean | worst | Δ floor | src | bstV | finV | zGr | zTe | zSp | pN | gGr | gTe | gSp | gEnt | dec% | rSp | lSp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| erm (floor .05) | 0.588 | 0.503 | — | 0.755 | 0.769 | 0.696 | 0.594 | 0.601 | 0.527 | 0.578 | 0.265 | 0.208 | 0.527 | 0.868 | — | — | — |
| dec-only s50 | 0.596 | 0.503 | +0.008 | 0.744 | 0.776 | 0.693 | 0.608 | 0.590 | 0.514 | 0.593 | 0.309 | 0.236 | 0.455 | 0.914 | 5.7% | 1.536 | 0.0000 |
| **dec-only s100** | **0.604** | 0.502 | +0.016 | 0.727 | 0.776 | 0.694 | 0.608 | 0.600 | 0.517 | 0.601 | 0.303 | 0.233 | 0.464 | 0.912 | 9.1% | 1.561 | 0.0000 |
| spatialCMI .003 | 0.603 | 0.503 | +0.015 | 0.734 | 0.776 | 0.692 | 0.612 | 0.600 | 0.518 | 0.600 | 0.299 | 0.233 | 0.468 | 0.909 | 5.3% | 0.919 | 0.0028 |
| all-CMI g/n/s | 0.599 | 0.503 | +0.011 | 0.723 | 0.776 | 0.694 | 0.609 | 0.597 | 0.514 | 0.597 | 0.312 | 0.237 | 0.451 | 0.927 | 5.2% | 0.927 | 0.0028 |

## 3. BNCI2014_001 — CSP-decodable subset {1,3,8,9} (the decisive cut)

Subjects where CSP decodes the 4-class signal. **Every CMI/decoder config REGRESSES here.**

| config | subset mean | worst | Δ floor | s1 | s3 | s8 | s9 |
|---|---|---|---|---|---|---|---|
| erm (floor .05) | **0.466** | 0.436 | — | 0.436 | 0.476 | 0.516 | 0.436 |
| dec-only s50 | 0.452 | 0.373 | −0.013 | 0.479 | 0.474 | 0.483 | 0.373 |
| dec-only s100 | 0.448 | 0.366 | −0.018 | 0.476 | 0.460 | 0.490 | 0.366 |
| spatialCMI .003 | 0.450 | 0.368 | −0.016 | 0.481 | 0.465 | 0.484 | 0.368 |
| all-CMI g/n/s | 0.432 | 0.372 | **−0.033** | 0.394 | 0.472 | 0.491 | 0.372 |

Non-decodable subset {2,4,5,6,7} (erm 0.278 ≈ 4-class chance 0.25): every config **gains** here
(+0.018…+0.033). So the full-2a "+0.007" for spatialCMI decomposes to **−0.71pp on decodable folds +
+1.37pp on chance folds** — the gain is logit reshuffling inside the chance band, and the model gets
**worse exactly where the task is decodable**. This pattern is uniform across all four configs.

## 4. Baseline note (no conflation in this table)

The in-screening `erm:0` carries fusion_floor=0.05 (2a 0.361 / 2015 0.588). The **prior** FBCSP-LGG ERM
reference is floor=0.0 (2a 0.349 / 2015 0.608). They differ by **+0.0125 on 2a but −0.0199 on 2015
(opposite signs)** — this is the fusion_floor effect alone, reported separately here and kept **out** of the
CMI verdict. All Δ columns above are CMI-vs-floor (floor held fixed), the only apples-to-apples baseline.

## 5. Verdict — NULL / NEGATIVE screen (recommend NO seeds 1/2)

- The R1 full-mean "pass" (dec50/dec100/spatialCMI improve both means; allCMI fails 2a) is arithmetically
  real but **mechanism-falsified**: it reverses on every CSP-decodable subject and lives entirely in the
  chance band. **R2 (improve CSP-decodable while preserving 2015) is met by NO config.**
- **All deltas are inside single-seed noise**: paired-fold |t| < 1.3 everywhere; sign tests are coin-flips;
  per-fold delta SD (~0.035–0.042) is 5–10× the celebrated deltas. Toggling fusion_floor 0→0.05 moved the
  **control itself** by more than any config's gain (+1.2pp 2a / −2.0pp 2015).
- **Worst-fold robustness flat or worse**; 2015 pushes more folds to chance (near-chance <0.51: erm 1/12 →
  every CMI config 3/12).
- The **actual method thesis — all-CMI (graph+node+spatial) — is the single clearest loser** (only 2a
  full-mean regressor; worst on the decodable subset, 0/4 wins).
- **Guards all clean** (dec% 5–12% < 20%; reg_spatial_gls finite 0.92–1.63; loss_spatial active only for
  λspatial>0; fusion_floor recovered gate_graph 0.27→0.31–0.34 without destroying zero_spatial). Clean
  machinery + null effect = a validated pipeline with **no method signal** — the textbook negative screen.

**Next step is architecture / fusion work, not more seeds.** Spending two more full-LOSO GPU seeds would
be chasing a chance-band artifact.

### Narrow PI-override fallback (NOT the recommendation)
If the PI wants one cheap confirmation despite the null read, promote **only** spatialCMI
(`fbdualpc:0.000:0.000:0.003:0.000:0.100:50`) — the sole method-carrying config net-positive on both full
means, only winning 2a fold record (5W/4L), only worst-fold beating erm, clean guards. **Seeds 1/2 must
pre-register the CSP-decodable {1,3,8,9} subset as the PRIMARY endpoint**: if the −0.013…−0.018 decodable
regression holds, the full-mean gain is confirmed a chance-fold artifact and the config dies. Do **not**
advance all-CMI (regresses where it matters) or dec100 (lead is one-subject-driven, highest decoder burden).

*Screening only; method judgment requires full-LOSO × multiple seeds. PI-gated decision.*
