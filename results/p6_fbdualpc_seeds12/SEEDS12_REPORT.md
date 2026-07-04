# P6 seeds 1/2 confirmation — pre-registered 3-seed verdict

Config = spatialCMI `fbdualpc:0.000:0.000:0.003:0.000:0.100:50` vs `erm:0`, both `fusion_floor=0.05`, full-LOSO.
Seeds: 0 (from `p6_fbdualpc_full_s0`), 1, 2 (from `p6_fbdualpc_seeds12`). 42/42 folds, 0 NaN. Pre-registration:
`5c26e6d`. seed0 numbers reproduce the verified P6 aggregate exactly; seeds1/2 independently recomputed +
subject-map asserted.

## 3-seed table (Δ = spatialCMI − erm)

### BNCI2014_001 (2a)
| seed | erm_full | spc_full | Δ_full | erm_dec {1,3,8,9} | spc_dec | Δ_dec |
|---|---|---|---|---|---|---|
| 0 | 0.3615 | 0.3681 | +0.0066 | 0.4657 | 0.4497 | −0.0161 |
| 1 | 0.3692 | 0.3601 | −0.0091 | 0.4683 | 0.4609 | −0.0074 |
| 2 | 0.3356 | 0.3470 | +0.0114 | 0.3837 | 0.4319 | +0.0482 |
| **3-seed mean** | | | **+0.0030** (sd .0087) | | | **+0.0082** (sd **.0285**) |

### BNCI2015_001 (2015)
| seed | erm_full | spc_full | Δ_full |
|---|---|---|---|
| 0 | 0.5881 | 0.6028 | +0.0147 |
| 1 | 0.6108 | 0.6119 | +0.0011 |
| 2 | 0.6125 | 0.6137 | +0.0012 |
| **3-seed mean** | | | **+0.0057** (sd .0064) |

## Pre-registered verdict: **SURVIVE (by the letter of the rule)**

PRIMARY 2a CSP-decodable Δ = **+0.0082 ≥ 0**; SECONDARY 2015 full Δ = **+0.0057 ≥ 0**. The rule (5c26e6d) says
SURVIVE requires primary ≥ 0 AND 2015 ≥ 0. Both hold → **SURVIVE**. Reported as pre-registered — no post-hoc
goalpost-moving.

## …but the pass is FRAGILE / artifact-driven (full disclosure)

1. **Entirely seed2-driven, and 2/3 seeds (incl. the pre-registered seed0) are NEGATIVE.** Per-seed decodable
   Δ = −0.0161 / −0.0074 / **+0.0482**. The across-seed SD (0.0285) is **3.5× the mean** (+0.0082) — the mean
   is not distinguishable from 0.
2. **The +0.0482 on seed2 is a collapsed-ERM-baseline artifact, not a spatialCMI gain.** seed2 ERM on the
   decodable subjects cratered — s8/s9 = **0.307 / 0.283** (≈ 4-class chance 0.25) vs seed0/1 s8/s9 =
   0.516/0.436 and 0.533/0.354. spatialCMI's **absolute** decodable performance `spc_dec` =
   0.4497 / 0.4609 / **0.4319** is flat-to-declining and **never beats ERM's best (0.4683)**. The positive
   mean comes from ERM falling on a bad seed, not spatialCMI rising.
3. **2015 gain is seed0-only.** +0.0147 (seed0) → ~0 on seeds1/2 (+0.0011, +0.0012).
4. **One genuinely consistent positive:** s1 improves on all 3 seeds (+0.045/+0.004/+0.061, mean +0.037); the
   other three decodable subjects are mixed/noisy (s3 −0.019, s8 −0.008, s9 +0.024, all with large seed swings).

## Honest read

The letter of the pre-registered rule says SURVIVE, but the evidence does **not** credibly show spatialCMI
helps the CSP-decodable subjects: the pass rests on ERM baseline instability on one seed, spatialCMI's absolute
decodable accuracy is flat-to-declining, and 2 of 3 seeds regress. This is consistent with the P6 seed0
"null / chance-artifact" reading, now with the added nuance that the 3-seed *mean* crosses zero only via
baseline collapse. Per the pre-reg, SURVIVE = "a live candidate worth a wider sweep"; the honest expectation is
that a wider sweep confirms null. **PI-gated:** whether the fragile pass warrants a wider spatialCMI sweep, or
should be treated as effectively null with GPU focused on the P7a SOTA-track, is the PI's call.
