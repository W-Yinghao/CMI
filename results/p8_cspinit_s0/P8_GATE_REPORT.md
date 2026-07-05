# CIGL_52 P8 gate — CSP-init + spatial-aux, full-LOSO × 3 seeds × 4 variants → **PASS (verified)**

252 folds (21 × 4 variants × 3 seeds), 0 NaN, floor 0.05, worktree `8128cb0`. Numbers independently recomputed
to 3 decimals + adversarially verified (recompute / skeptic / advocate / data-quality). **Firewall CLEAN
126/126** source_csp folds (every B/D fold: `csp_excluded_target=True`, source-val subject excluded from
`csp_fit_subjects`).

Variants: **A** random/aux0 (baseline) · **B** csp/aux0 · **C** random/aux0.2 · **D** csp/aux0.2.

## 2a CSP-decodable {1,3,8,9} — Δ vs baseline A (PRIMARY, need ≥ +0.02)

| variant | 3-seed mean Δdec | per-seed | sd | Δ2a_full | Δ2015 |
|---|---|---|---|---|---|
| B csp/aux0 | +0.062 | +.027 / +.059 / +.099 | .029 | +0.032 | +0.022 |
| C random/aux0.2 | +0.028 | +.024 / +.021 / +.039 | .008 | +0.015 | −0.004 |
| **D csp/aux0.2** | **+0.096** | +.081 / +.062 / +.145 | .035 | **+0.051** | **+0.019** |

**All 9 variant×seed cells positive.** D per-subject seed0: s1 .385→.458, s3 .509→.568, s8 .505→.592,
s9 .339→.443 (all up). Per-subject across seeds: 11/12 decodable cells up, 1 tie, **0 regressions**.

## Verdict: PASS — the first genuine positive in this line (P6 null → P7a FAIL → **P8 PASS**)

All pre-committed criteria met by D: PRIMARY +0.096 ≥ +0.02 **and ≥ +0.02 on every seed** (min +0.062);
SECONDARY 2a-full +0.051 ≥ 0, 2015 +0.019 ≥ −0.01 (both positive every seed); **no memorization** (D
source_bacc 0.55–0.86, never ~1.0, ≈ baseline A's); **spatial load-bearing** (removing it costs D ~0.131 vs A
~0.071 on 2a-full, ~2×); gate opens to spatial (D 0.60 vs A 0.42). This is NOT P6's fragile,
baseline-collapse-driven, 2/3-negative pass — here the effect is stable across seeds, on genuinely-decodable
subjects (A≈0.42 ≫ 0.25 chance), near-zero on non-decodable folds (CSP-spatial signature).

## Three honest discounts (carry these — do not over-claim)

1. **Report the effect as ~+0.06–0.07, not +0.096.** The headline is inflated ~+0.03 by a seed2 **baseline-A
   collapse** (A fell to 4-class chance on s8/s9 at seed2 — the P6 failure mode, partially present). But the
   effect survives it: drop-collapsed-cells +0.064, leave-seed2-out +0.071, worst-seed +0.062 — all ≥ 3× the
   gate. Collapse inflates the magnitude; it does not manufacture the effect.
2. **CSP-init is the load-bearing ingredient; the spatial-aux is a small, weakly-supported increment.** B
   (csp-only) alone = +0.062; D−B = +0.034 3-seed but only +0.003 at seed1. The two knobs are ~additive on
   average, but if a claim rests specifically on the aux mechanism it is thin/seed-fragile. **B (CSP-init only)
   is the conservative fallback** if the aux does not hold on forward runs.
3. **Scope is small & single-run:** n=4 decodable subjects, D absolute decodable ≈ 0.515 (an *improved weak*
   4-class decoder, not a strong one), one site, one seed-triple. A promising candidate, not deployable-proven.

## Recommendation (PI-gated)

Promote **D** (csp+aux) forward as the live candidate, publish the **collapse-robust +0.06–0.07** as the effect
size, and treat **CSP-init as the load-bearing mechanism** with spatial-aux as a smaller, still-to-confirm
increment. Multi-seed is already in hand (seeds 0/1/2 done). Natural next steps for the PI to gate: confirm on a
held-out **site/dataset** and/or additional seeds before calling it deployable; decide D-vs-B as the headline
variant. No P7b / adaptive-gating / cov_tangent revival — this is a cleaner lever than any of them.
