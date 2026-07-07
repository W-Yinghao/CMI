# Target-information frontier (Fork 1) — FINAL VERDICT (PARKED with finding)

Branch `science-target-info-v1`. Semi-synthetic; NOT a final paper claim. Jobs: v0 smoke 888028, hardened v1
888437, label-budget frontier 888470. Commits: v0 5c4d6d6, hardening 7e7e4a3, v1 9c87377, frontier 56b449e.
Fork 1 is CLOSED here; sharper-estimator R&D is future work, not pursued now.

## 1. Question
> How much target information is needed to safely cross the source-only acceptance ceiling?

## 2. v0 smoke (weak / bootstrap estimator)
A bootstrap target-calibration LCB produced deployable accepts that rose with k, but the false-accept rate was
too high (~23% overall; 100% at k=1). This showed the target-label SIGNAL exists but the estimator was UNSAFE.

## 3. Hardened finite-sample rerun (v1)
Replaced the bootstrap with a stratified bounded lower confidence bound: Maurer-Pontil empirical-Bernstein on the
paired per-trial balanced-accuracy difference (range [-1,1]), Bonferroni over (classes x candidate interventions),
underpowered classes -> abstain (never a point estimate).
```
false accepts        = 0
deployable true accepts = 0
k<=16 labels/class cannot certify
```

## 4. Label-budget frontier + oracle finding
Extended k to {1,2,4,8,16,24,32,40,50} (= the full 50/class calibration budget), estimator unchanged, same data.
```
0 deployable accepts at EVERY k up to 50, both worlds (v2_source_invisible + source_rich_source_visible)
false-accept rate 0.000 ; harmful 0
bounded cal-LCB rises monotonically with k (clipped: ~-1.0 at small k -> -0.53/-0.59 at k=50) but never nears +0.01
B4 oracle (all target labels; DIAGNOSTIC) CONFIRMS target-beneficial cells exist: audit ΔbAcc mean +0.018
  (v2) / +0.021 (source_rich), max +0.080
```
Therefore the failure is **certification power, not absence of effect**. Extrapolating the bound width
(~7R ln(2/δ)/(3(n-1))), certifying even the strongest ~+0.1 cell needs ~340 labels/class; the typical ~+0.02
effect needs far more -- an order of magnitude beyond the <=50/class this EEG calibration setting provides.

## 5. Final interpretation
> Few-shot target labels do not safely turn refusal into acceptance under a distribution-valid finite-sample
> certificate. Crossing the ceiling likely requires either much larger target-label budgets or additional
> modeling assumptions.

Correct framing: **target information helps in principle (the oracle sees the benefit), but SAFE certification of
small effects is sample-complexity limited at realistic calibration budgets.** This empirically instantiates
CEILING_THEORY Proposition 3 (target-label sample complexity n=O(ε^-2 log(1/δ)) per class).

## 6. Future R&D (NOT current result)
- sharper structure-exploiting estimator (variance-reduction / paired low-variance)
- model-assisted target-risk bounds
- active calibration / sequential alpha-spending with stronger structure
- a dedicated target-information budget paper
All marked FUTURE WORK.

## Reporting note (bounded LCB range)
The raw empirical-Bernstein lower bound is valid even below -1 (outside the natural [-1,1] balanced-accuracy-
difference range) at tiny k; for readability the report tables/figures use the metric-range-CLIPPED LCB, while the
summary JSON retains the raw (unclipped) diagnostic value (`cal_lcb_max_raw_unclipped`).

## Overall project arc after Fork 1
```
Real EEG:  source-fitted erasure does not improve target bAcc.
Track B:   source-only gate rejects harmful/useless erasures.
V2:        source-only acceptance CEILING (target-beneficial shift can be source-uncertifiable).
Fork 2:    source-rich environments give a source-only positive on low-dim EEGNet, but discovery is
           representation-dependent / fragile (fails on high-dim TSMNet).
Fork 1:    target labels reveal target-beneficial cells, but safe few-shot certification is sample-complexity
           limited (0 safe accepts up to k=50 under a sound bound; oracle confirms the effect is real).
```
Not "nothing works": strict source-only selective invariance should be **refusal-first**; moving from refusal to
control needs either source-rich environments that actually cover the shift, or enough target information /
assumptions to certify benefit -- both routes have nontrivial limits. Manuscript stays frozen (2a-only PDF stale).
See [[tos-cmi-method-deepen-v2]], `SOURCE_RICH_FINAL_VERDICT.md`, `CEILING_THEORY.md`.
