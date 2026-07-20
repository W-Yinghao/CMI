# C31 Red-Team Verification — Endpoint Geometry (multi-agent adversarial pass)

Four independent adversarial verifiers (ultracode Workflow, `general-purpose` agents, opus-4-8-1m) re-attacked the
five C31 endpoint-geometry verdicts against the REAL in-regime data (3804 candidates, 9 targets, 54 trajectories =
9 targets × 3 seeds × 2 levels). Each verifier loaded the artifacts exactly as C31 does and tried to REFUTE its
assigned claim with a fresh permutation / cluster-bootstrap / confound test. A leg is kept only if it survives the
primary `IMPROVE_MARGIN=1e-9` definition, a 9-target cluster-bootstrap, and a by-construction check.

## Summary

| Endpoint verdict | Prior | Reconciled | What changed |
|---|---|---|---|
| **E1** accuracy–calibration trade-off | FALSIFIED | **HOLDS (falsified, robust)** | survived all attacks incl. shared-ERM (provably inert) |
| **E8** Pareto wall | FALSIFIED | **HOLDS (falsified)** + caveat | headline 3.7% is OR-calibration-favorable; strict → 22–35% but still sub-majority |
| **E3** joint common + within-visible + pooled gauge-broken | (new) | **DOWNGRADE (wording)** | "pooled ≈ chance" too strong → **collapsed / non-deployable** |
| **E7** general per-target gauge offset | (new) | **HOLDS** | directly evidenced by the E3 gauge decomposition |
| **E4** source rank *accuracy-specific* | (new) | **DOWNGRADE** | → *accuracy-aligned-BY-CONSTRUCTION, endpoint-nonspecific within noise (except vs ECE)* |

## The headline question — is E4 real? **No. DOWNGRADE.**

1. **Not distinguishable from calibration within per-target noise (primary margin).** Reproduced within-target rank
   strengths: accuracy_good **0.159**, calibration_good **0.120**, nll_good 0.113, ece_good 0.066. A 9-target paired
   cluster-bootstrap (`RandomState(0)`, per-target AUCs resampled with replacement):
   - accuracy − calibration_good: gap **0.039**, 95% CI **[−0.029, 0.100]**, P(gap>0)=0.87 → **includes 0**
   - accuracy − nll_good: gap 0.046, CI **[−0.010, 0.103]** → includes 0
   - accuracy − ece_good: gap **0.093**, CI **[0.012, 0.159]** → **excludes 0** (the *only* surviving contrast)
   The gap spans only 9 targets and is dominated by one calibration outlier (target 2 AUC 0.87 vs a ~0.55 cluster).
   The original C31 write-up also mislabeled its own numbers — the "0.113" it attributed to *calibration_good* is
   actually *nll_good*; true calibration_good strength is 0.120.
2. **Specificity is substantially by-construction.** The frozen probe score's training label IS accuracy_good —
   identical for all 3804 rows (0 mismatches) — so "ranks accuracy best" is tautological with the objective; and
   89.7% of accuracy-good candidates are also calibration-good (φ=0.43), so the calibration ranking is largely
   *inherited*. The downgrade REINFORCES the "same object" reading (an accuracy-specific rank would have *split*
   accuracy from calibration; it does not).

## Per-endpoint detail

**E1 (no trade-off) — HOLDS, robustly falsified.** corr(bacc_delta, nll_improve)=**+0.590**, ece **+0.605**;
epoch-residualized +0.504/+0.554. Shared-ERM confound is *provably inert*: the ERM reference is exactly constant
within each (seed,target,level) trajectory (max spread 0), absorbed by the residualization intercept — RAW
corr(bacc,−nll)/corr(bacc,−ece) with no ERM subtraction reproduce +0.504/+0.554 identically. Target bootstrap: mean
+0.523, 95% CI **[0.340, 0.739]**, 100% of reps >0; **9/9** targets individually positive (heterogeneous magnitude
+0.14…+0.86). accuracy_good_calibration_bad 0.049 (0.019 robust); calibration_good_accuracy_flat 0.259 — the
opposite of a trade-off.

**E3 — DOWNGRADE (wording only; mechanism strengthened).** joint_good rate **0.424** (exact); within_target_auc
**0.672** outside a within-target permutation null (p=0.002), **sign-consistent 9/9** (0.55–0.81, no C30-style sign
flip). Gauge mechanism confirmed: same-target pair concordance 0.678 (10% of pairs) vs cross-target 0.525 (90%)
reconstructs pooled exactly; the per-target gauge offset is *anti-aligned* (location-only null 0.439 < 0.5); oracle
per-target z-calibration (target-specific, source-unobservable) recovers pooled to **0.651**. **But** pooled 0.541 is
NOT literally at chance at the primary margin — it is *outside* a global-shuffle null (|auc−0.5|=0.041 vs null 95th
0.018, max 0.030; p=0.002, 0/500 across seeds 0/1/7/42) because within-target rank leaks into the pool. It reaches
literal chance (0.489, inside null) only at the 0.02 margin. → state pooled as **"collapsed / non-deployable"**.

**E7 — HOLDS.** The general per-target gauge offset breaking pooled transport for *every* endpoint is directly
evidenced by the E3 decomposition (anti-aligned location null 0.439; oracle recalibration 0.541→0.651) and by E8
(same gauge failure for accuracy, calibration, joint). Same rank-vs-gauge split as C22–C30, now endpoint-general.

**E8 (no Pareto wall) — HOLDS + definitional caveat.** accuracy-oracle calibration-bad **3.7%** (2/54) primary /
11.1% (6/54) robust; cluster-bootstrap CIs [0, 9.3%] and [3.7%, 20.4%], both far below the 0.50 wall threshold.
Taxonomy returns E3/E7 (never E8/E1) under both margins; bAcc↔calibration co-improve +0.60. **Caveat:** 3.7% uses the
frozen OR-calibration; strict both-NLL-and-ECE → 22% primary / 35% robust, still SUB-MAJORITY (no wall). The
trade-off that *does* exist is ASYMMETRIC: the calibration-oracle is accuracy-flat 17–46% — a base-rate effect
(calibration-good 0.68 > accuracy-good 0.47) that MATCHES C16's calibration-improved/accuracy-flat outcome.

## Bottom line — survives?

**Yes, at the within-target / diagnostic level (NOT deployable).** accuracy rank, calibration rank, and the joint
Pareto point are largely ONE within-target-rankable object: they improve together (E1, +0.60, 9/9), are ordered
indistinguishably by the source score (E4-downgraded, gap CI includes 0), and the joint set is common (E3/E8). The
E4 downgrade STRENGTHENS this (it removes the one claim that argued accuracy and calibration were different).
Honest boundaries: (a) the object's pooled/cross-target transport is gauge-broken and non-deployable (E3 collapsed,
E7 anti-aligned offset); (b) "largely," not identical — base-rate asymmetry + ECE as a partial exception; (c) this
RECONCILES C16 (source-observability/gauge failure), it does not overturn it. All oracles are diagnostic-only and
non-deployable; no endpoint or Pareto selector is claimed.

*(4 attack agents + 1 synthesis, 212,595 subagent tokens, 52 tool calls, run `wf_7b204661-de2`.)*
