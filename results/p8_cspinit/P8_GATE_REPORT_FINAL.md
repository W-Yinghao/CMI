# P8 FINAL — CSP-initialized FBCSP-LGG (+ spatial auxiliary): PASS, promoted

Paper-ready record of the P8 gate (raw evidence `../p8_cspinit_s0/`, verified commit `ad066ba`). Numbers are
3-seed means over full-LOSO seeds 0/1/2 × 4 variants × 21 folds (252 folds, 0 NaN, floor 0.05), independently
recomputed + 4-lens adversarially verified; source-CSP firewall clean 126/126.

**Headline (conservative, use this):**
> On BNCI2014_001 CSP-decodable subjects {1,3,8,9}, the promoted model **D (source-CSP init + spatial aux)**
> improves balanced accuracy by a **collapse-robust ≈ +0.06–0.07** over the random-init baseline. **B (CSP-init
> only) alone gives +0.062**, showing source-CSP initialization is the load-bearing mechanism; the spatial
> auxiliary adds a smaller, less stable increment.

Do **not** report "+0.096, full stop": the raw 3-seed mean is inflated ~+0.03 by a seed2 baseline collapse.
Collapse-robust figures: drop-collapsed-cells **+0.064**, leave-seed2-out **+0.071**, worst-seed **+0.062** —
all ≥ 3× the pre-committed +0.02 gate.

Variants: **A** random/aux0 (baseline) · **B** csp/aux0 · **C** random/aux0.2 · **D** csp/aux0.2.

## Table 1 — Variant aggregate (3-seed mean, Δ vs A)
| variant | 2a decodable | Δdec | 2a full | Δfull | 2015 full | Δ2015 |
|---|---|---|---|---|---|---|
| A random/aux0 | 0.4197 | — | 0.3441 | — | 0.6094 | — |
| B csp/aux0 | 0.4813 | **+0.062** | 0.3760 | +0.032 | 0.6318 | +0.022 |
| C random/aux0.2 | 0.4475 | +0.028 | 0.3591 | +0.015 | 0.6054 | −0.004 |
| **D csp/aux0.2** | **0.5153** | **+0.096** (robust +0.06–0.07) | 0.3956 | +0.051 | 0.6285 | +0.019 |

## Table 2 — Seed robustness (Δdec vs A, per seed)
| variant | seed0 | seed1 | seed2 | mean | sd |
|---|---|---|---|---|---|
| B | +0.027 | +0.059 | +0.099 | +0.062 | .029 |
| C | +0.024 | +0.021 | +0.039 | +0.028 | .008 |
| D | +0.081 | +0.062 | +0.145 | +0.096 | .035 |

Every variant, every seed, positive on the decodable subset (worst cell +0.021). Not P6-fragile.

## Table 3 — Subject-level (2a, 3-seed mean bAcc; decodable = the endpoint, hard = reported separately)
| subject | class | A | B | C | D | D−A |
|---|---|---|---|---|---|---|
| subj1 | decodable | 0.458 | 0.497 | 0.480 | 0.489 | +0.031 |
| subj3 | decodable | 0.464 | 0.539 | 0.500 | 0.560 | +0.097 |
| subj8 | decodable | 0.439 | 0.462 | 0.452 | 0.560 | +0.122 |
| subj9 | decodable | 0.318 | 0.427 | 0.358 | 0.452 | +0.134 |
| subj2 | hard | 0.270 | 0.276 | 0.294 | 0.292 | +0.023 |
| subj4 | hard | 0.314 | 0.329 | 0.351 | 0.326 | +0.012 |
| subj5 | hard | 0.262 | 0.257 | 0.256 | 0.251 | −0.012 |
| subj6 | hard | 0.311 | 0.329 | 0.273 | 0.328 | +0.017 |
| subj7 | hard | 0.261 | 0.268 | 0.269 | 0.302 | +0.041 |

Gain concentrates on genuinely-decodable subjects (all 4 up); hard subjects (≈ chance 0.25) barely move — the
CSP-spatial signature, not a uniform shift.

## Table 4 — Mechanism / decomposition (3-seed mean)
| effect | 2a decodable | 2a full | 2015 |
|---|---|---|---|
| CSP-init (B−A) | **+0.062** | +0.032 | +0.022 |
| Aux (C−A) | +0.028 | +0.015 | −0.004 |
| Combined (D−A) | +0.096 | +0.051 | +0.019 |
| Aux on top of CSP (D−B) | +0.034 (seed1 **+0.003**) | +0.020 | −0.003 |
| CSP on top of Aux (D−C) | **+0.068** | +0.037 | +0.023 |

Diagnostics (D vs A, 3-seed): gate_spatial 0.60 vs 0.42 (branch used more); zero_spatial removal cost 0.131 vs
0.071 (2× more load-bearing); source_bacc 0.603 vs 0.60-ish (no memorization); spatial_aux_target_bacc ≈0.41.

## Claim hierarchy (promoted)
- **Primary method candidate:** D (source-CSP init + spatial aux).
- **Primary mechanism claim:** source-CSP initialization (supported by B alone = +0.062, and CSP-on-top-of-aux
  D−C = +0.068).
- **Secondary mechanism claim:** the spatial auxiliary provides an additional but weaker, less stable gain
  (D−B = +0.034, ~0 at seed1). **B is the conservative fallback** if the aux does not hold.

## Honest discounts (carry always)
1. Effect = **+0.06–0.07** collapse-robust, not +0.096 (seed2 baseline-A collapse inflates the raw mean).
2. CSP-init is the mechanism; spatial-aux is a smaller, seed-fragile add-on.
3. Small scope: n=4 decodable subjects, D absolute decodable ≈0.515 (an *improved weak* 4-class decoder),
   single site, 3 seeds. Promising candidate, not deployable-proven — needs external/site confirmation + strong
   baselines.
