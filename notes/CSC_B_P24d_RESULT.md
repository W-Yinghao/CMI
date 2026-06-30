# CSC Route B3-P2.4d — cross-budget α-spending: the first CLEAN-control config (power caveats remain)

DEVELOPMENT diagnostic, simulator-only. Method LOCKED (P2.4c: pc_centered + fixed-margin null + studentized
subject-consistency gate); the ONLY change is α_budget = α_family/n_decision_budgets = 0.05/2 = **0.025** on
both p-gates AND the LCB at 1−α_budget = **0.975** (deterministic Bonferroni across the m=20/30 positive
budgets; no sweep). FRESH independent seeds (controls 3000, power 4000). Artifact `csc/results/b3_p24d.json`
(SLURM 877663). Red-team-verified. **No finite-sample claim; NO freeze/confirmatory/real-EEG.**

## Control false-confirm — per budget (the stratified view), FRESH seeds

| control kind | m=20 (CP-up) | m=30 (CP-up) |
|---|---|---|
| `missing_pair` / `unequal_epochs_extreme` | 0 (0.010) | 0 (0.010) |
| `clean` | **0 (0.010)** | **0 (0.010)** |
| `paired_covariate` | 0.0035 (0.016) | 0.0069 (0.022) |
| `paired_covariate_plus_label` | 0 (0.010) | 0.0069 (0.022) |
| `paired_label` | 0.0069 (0.022) | 0 (0.010) |
| `random_label` | 0 (0.010) | **0.0243 (0.045)** |
| **pooled (m≥20)** | — | **14/4032 = 0.0035 (CP-up 0.005)** |

**Every control kind is ≤ α at BOTH budgets, point AND CP-upper** (max CP-up = `random_label`@m30 0.045 < α).
`clean`@m30 (the P2.4c residual 0.052) is **0**. NO hard-fail flags; **0 sampler failures**.

## Power (FRESH seeds) — and its caveats

| positive | m20 pooled | m30 pooled | by scenario @ m20 |
|---|---|---|---|
| `paired_concept` | 0.799 | 0.819 | **1.00** in baseline/high_nuisance/high_subject_tau/imbalanced; **`label_noise` 0.50**; `few_epochs` 0.29 (eligibility) |
| `paired_concept_plus_cov` | 0.806 | 0.826 | same pattern |
| `paired_pure_conditional` (secondary) | 0.146 | 0.250 | low |

α-spending removed (vs P2.4c rule on the SAME fresh data): **24 control + 4 PRIMARY concept + 24 secondary**
— so the correction itself costs little primary power (4 vs P2.4c's 10).

## Honest caveats (do NOT overclaim)

1. **`label_noise` concept power is weak and SEED-VARIABLE** — 0.50 here (fresh seeds) vs 0.83 in the
   P2.4c run (seeds 1000/2000). This swing is mostly **intrinsic** (concept under 10% label noise is hard at
   m=20), not the α-spending (which removed ~1 there). So primary power is 1.00 only in the **clean/moderate**
   stress scenarios; heavy label noise is a genuine weak regime.
2. **`pure_conditional` (secondary) is low** (0.25 @ m30) — the cumulative tightening (fixed-margin +
   studentized + α-spending) leaves the subtle invisible-relabel case largely out of reach.
3. **The clean controls are MOSTLY SEED, not the rule (red-team correction; matched-seed decomposition).**
   Using `old_p24c_decision` (the P2.4c rule replayed on the SAME fresh data), the α-spending RULE's effect
   on matched seeds is real but **modest**: all-controls pooled **0.0094 → 0.0035**, `random_label`
   **0.0243 → 0.0122** (both ~halved), `clean`@m30 0.0035 → 0.000 (**1 confirmation removed**). The dramatic
   headline `clean`@m30 **0.052 → 0** is **~93% seed, ~7% rule** — the P2.4c run's 0.052 was a seed-specific
   high; even the *old* rule on these fresh seeds gives 0.0035. So I must NOT attribute the clean controls to
   the α-spending; the honest statement is "the rule roughly halves the matched-seed control rate, and these
   fresh seeds happen to land at 0." Crucially, **on the P2.4c seeds (1000/2000) the OLD rule had `clean`@m30
   = 0.052 > α** — so control cleanliness is NOT yet shown stable across seeds.

## My read (red-team-corrected) — strong development result, NOT yet a freeze-candidate

P2.4d is the **strongest development result so far**: on fresh seeds every control kind is ≤α at both
budgets (pooled 0.0035, no hard-flags, 0 sampler failures), primary concept power is **1.00 in 4/6
scenarios**, and the α-spending costs almost no primary power (4 removals, all in `label_noise`). The
α-spending RULE genuinely tightens controls on matched seeds (all-controls 0.0094→0.0035; `random_label`
halved) — a real, correct improvement.

But it is **NOT a freeze-candidate yet**, for three honest reasons:
1. **Control cleanliness is not shown stable across seeds.** On the P2.4c seeds the OLD rule had `clean`@m30
   = 0.052 > α; the rule brings that to ≤α on *point* (replay 0.035, CP-up 0.058 grazing), and the fresh
   run lands at 0 — but most of the fresh "0" is seed luck (caveat 3). A single fresh-seed pass is not
   enough to claim "controls ≤α"; that needs a **second independent seed** under the locked rule.
2. **Primary power is non-uniform** — `label_noise` concept CP-lower is **0.32** (and seed-variable
   0.50–0.83 across runs); `few_epochs` is eligibility abstention (disclose as such); `pure_conditional`
   secondary 0.25.
3. The artifact self-labels **`development_only: True` / "NOT error control."** Calling it a freeze would
   contradict its own classification.

**Recommendation:** do NOT freeze. Run **one more independent fresh-seed P2.4d control-resolution** (locked
rule, new seeds) to test whether the controls stay ≤α under the RULE rather than a lucky seed. If that
second seed is also clean, THEN draft the B3 freeze-candidate (primary = `concept`/`concept_plus_cov`
restricted to the 4 strong scenarios, pre-registered envelope excluding heavy-label-noise + few-epochs,
`pure_conditional` declared known-weak secondary), new tag `csc-b3-confirmatory-v1`. Still NO
freeze/confirmatory/real-EEG.
