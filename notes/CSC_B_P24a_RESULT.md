# CSC Route B3-P2.4a — control-resolution (48 clusters): residual is REAL label-composition leakage → no freeze

DEVELOPMENT diagnostic, simulator-only, `csc/mininfo/`. CONTROLS ONLY, method LOCKED
`pc_centered_calibrated` (no changes). 7 control kinds × 6 scenarios × m{0,20,30} × **48 fresh independent
clusters** (base_seed 1000). 48/cell is NOT error control (a 0/48 control still has CP-up ≈ 0.0605), but it
cleanly separates 24-cluster noise from method-level leakage. NO freeze/confirmatory/real-EEG. Artifact
`csc/results/b3_p24a_controls.json` (SLURM 877067). Numbers below independently re-aggregated from
per-cluster records (match the runner console exactly).

## Result — by-kind pooled false-confirm (decision budgets m∈{20,30}, 576 runs/kind)

| control kind | FC | rate | CP-upper | vs α=0.05 |
|---|---|---|---|---|
| `missing_pair` | 0/576 | 0.0000 | 0.0052 | **clean (guard holds)** |
| `unequal_epochs_extreme` | 0/576 | 0.0000 | 0.0052 | **clean (guard holds)** |
| `paired_covariate` | 11/576 | 0.0191 | 0.0314 | clean |
| `clean` | 24/576 | 0.0417 | 0.0581 | ≈α (was noise at 24 clusters) |
| `paired_covariate_plus_label` | 31/576 | 0.0538 | 0.0719 | **ABOVE α** |
| `paired_label` | 38/576 | 0.0660 | 0.0856 | **ABOVE α** |
| `random_label` | 42/576 | 0.0729 | 0.0933 | **ABOVE α** |
| **pooled all** | 146/4032 | 0.0362 | 0.0414 | conservative on average |

## Verdict — HARD-FAIL of the proceed-to-freeze criteria

The 48-cluster round answers the open question definitively:

- **The two safety guards are PERMANENT wins.** `missing_pair` and `unequal_epochs_extreme` are **0/576**
  (CP-up 0.005) — the P2.3 fail-closed breaches are robustly closed by the pair-integrity + eligibility
  guards.
- **The clean/covariate creep was NOISE.** At 48 clusters `clean` is 0.042 (≈α) and `paired_covariate`
  0.019 (below α) — the 24-cluster P2.4 elevation there does not reproduce.
- **The label-composition leakage is REAL.** `random_label` (0.073), `paired_label` (0.066) and
  `paired_covariate_plus_label` (0.054) are all **above α with CP-uppers 0.07–0.093**, and the elevation is
  **kind-concentrated across all 6 scenarios** (not a label_noise corner). Five hard-flags fired
  (random/paired_label/cov+label kind-concentration; clean/covariate repeated — the latter two now
  attributable to noise per the rates above, but the three label-kinds are genuine).

The sharpest single fact: **`random_label` false-confirms 7.3%.** Random labels carry no `P(Y|Z)` signal at
all, yet the calibrated cross-fitted test reports a conditional **concept change** 7.3% of the time → the
null is **anti-conservative for label/prior-composition differences between conditions**. The
class-balanced loss + cross-fit reduced this (P2.3 random_label was much worse) but did **not** remove it.

## Localization (what is and isn't fixed)

| component | status |
|---|---|
| pair-integrity guard (missing pairs) | **fixed, robust** (0/576) |
| eligibility guard (short records) | **fixed, robust** (0/576) |
| non-label controls (clean, covariate) | **clean** at 48 clusters |
| **label/prior-composition controls** | **STILL LEAK** (0.054–0.073 > α, kind-concentrated) |

So the remaining problem is precisely scoped: the test still lets **condition differences in label
composition / prior** masquerade as concept change. That is exactly what the pre-declared **P2.4b
prior/noise-robust null** targets.

## Recommendation — do NOT freeze; proceed to P2.4b (reviewer's pre-declared next step)

Per the reviewer's plan ("if 48-cluster controls prove label/prior or random-label leakage, then enter
P2.4b prior/noise-robust null"), this result triggers **P2.4b**. Concretely, the null must be made robust
to per-condition label-composition / prior differences — e.g. draw the bootstrap `Y*` under a
**condition-matched class prior** (so a pure prior/composition difference cannot inflate `T`), or add a
label-composition gate. Method otherwise stays LOCKED (`pc_centered_calibrated`); no rank/C/RFF/score-space
changes. After P2.4b, re-run this same 48-cluster control-resolution; freeze only if the label-composition
controls drop to ≤ α and the guards stay 0. Still NO freeze/confirmatory/real-EEG.
