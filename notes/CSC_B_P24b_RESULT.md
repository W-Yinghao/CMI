# CSC Route B3-P2.4b — condition-matched fixed-margin h0 bootstrap: EVIDENCE PACKAGE

DEVELOPMENT diagnostic, simulator-only, `csc/mininfo/`. Method LOCKED `pc_centered_calibrated`; ONLY the
null is the condition-matched fixed-margin h0 bootstrap. 48 control clusters/cell + 24 power clusters/cell,
6 scenarios, m{0,20,30}; decision budgets m∈{20,30}. NO finite-sample claim; NO freeze/confirmatory/real
EEG. Artifacts `csc/results/b3_p24b.json`, `csc/results/b3_p24b_sampler_audit.json` (SLURM 877392).
Presented as EVIDENCE (no pass/fail lead, per the reviewer); my read is at the end.

## 1. Control table — by-kind false-confirm (m≥20, 576 runs/kind): fixed-margin vs standard-null-would

| control kind | fixed-margin | CP-upper | standard-null-would | vs α=0.05 |
|---|---|---|---|---|
| `missing_pair` | 0/576 = 0.0000 | 0.005 | 0.0000 | clean (guard) |
| `unequal_epochs_extreme` | 0/576 = 0.0000 | 0.005 | 0.0000 | clean (guard) |
| `paired_covariate` | 11/576 = 0.0191 | 0.031 | 0.0260 | ok |
| `paired_covariate_plus_label` | 16/576 = 0.0278 | 0.042 | 0.0573 | **fixed (was >α-ish)** |
| `clean` | 19/576 = 0.0330 | 0.048 | 0.0451 | ≈α |
| `paired_label` | 22/576 = 0.0382 | 0.054 | 0.0729 | **fixed (≤α; CP-up 0.054)** |
| `random_label` | 38/576 = 0.0660 | 0.086 | 0.0729 | **STILL > α** |
| **pooled all** | 106/4032 = 0.0263 | 0.031 | 0.0392 | conservative |

## 2. Power table (fixed-margin null)

| positive | m20 (5/6 scen) | few_epochs m20 | m30 pooled |
|---|---|---|---|
| `paired_concept` | **1.00** | 0.33 (eligibility) | 0.889 |
| `paired_concept_plus_cov` | **1.00** | 0.33 (eligibility) | 0.889 |
| `paired_pure_conditional` | — | 0.00 | 0.451 (secondary) |

`few_epochs` `paired_concept` @ m20 = 8 CONCEPT_CONFIRMED / 16 NEED_MORE_LABELS (eligibility abstention,
invalid_pair_rate 0 — honest, not silent).

## 3. Standard-null-would vs fixed-margin (what the fixed margins bought)

The fixed-margin null lowers the label-composition controls relative to the standard null:
`paired_label` 0.073→0.038, `paired_covariate_plus_label` 0.057→0.028, pooled 0.039→0.026. It does **not**
move `random_label` (0.073→0.066) — because `random_label`'s per-condition class margins are already
balanced, so holding them fixed removes nothing. (This is the `would_confirm_under_standard_null` vs
fixed-margin per-cluster comparison; the `PRIOR_COMPOSITION_MATCHED_NULL_NOT_SIG` gate reason fires on the
cells the fixed margins rescued.)

## 4. Sampler diagnostics (audit; no decision change)

- **Mixing:** acceptance 0.35–0.91 (random_label high since p0≈uniform; concept/label 0.35–0.49);
  changed-label fraction 0.35–0.66 (chain moves off the observed start).
- **n_swaps sensitivity** (base × {1,5,10}): max fixed-margin p-range **0.132**, on far-from-α cells; no
  decision flips (concept p=0.007 stable; near-α `paired_label` range ≤0.007).
- **Generator** (full-audit vs fold-local h0): max |Δp| **0.119**, again far-from-α; near-α difference
  ≤0.04 (full-audit is the declared primary).
- **Sampler health:** margin_preserved True on every cell that ran the sampler; total sampler failures 0.
- **Unit:** `margin_unit = label_unit = TRIAL`; acquisition = query m subjects, label all their epochs;
  aggregation = subject-condition vote.

## Red-team verification (independent re-aggregation of all 4032 control + 1296 power runs)

An independent agent reproduced every number bit-for-bit and **found no false statements**. Refinements it
added (all toward MORE caution): (i) `paired_label`'s point rate 0.038 is ≤α but its **CP-upper 0.054 still
grazes α** → "fixed in expectation, not certified clean"; (ii) `random_label`'s elevation is concentrated
across **all 6** scenarios (this note's earlier flag list named 5 — an *under*-statement); (iii)
`random_label`'s one-sided binomial P(rate>α) = 0.053 is **borderline** (does not formally reject H0:p≤α at
0.05), yet the point 0.066 + CP-up 0.086 mean calling it a fail is the **conservative, correct** direction;
(iv) per-cell n is **48** (6 scenarios × m{20,30} × 48 clusters = 576/kind); (v) the "different failure
mode" framing for `random_label` is an **interpretation/hypothesis**, not proven by these artifacts.
Sampler correctness: margin_preserved True on all 3744 records that ran the sampler (short-circuit cells
are None), **0 sampler failures, 0 bootstrap-invalid**; audit n_swaps/generator variation moved **no**
near-α decision.

## My read (the reviewer judges)

The fixed-margin null is the **correct fix for label-composition / prior-shift leakage** — it drops
`paired_label` and `paired_covariate_plus_label` to ≤α and lowers the pooled control rate, with **concept
power fully retained** (1.00 ex-`few_epochs`) and the **guards permanent** (missing_pair / unequal_epochs
0/576). It is **not a clean pass**, because **`random_label` remains above α (0.066, CP-up 0.086,
kind-concentrated across all scenarios)**. That residual is a *different* failure mode: with balanced
margins there is nothing for the fixed-margin null to hold fixed, so it reflects **cross-fit-T
over-rejection on pure-noise labels**, not label composition — which the null cannot fix.

Per the pre-registered promotion criterion ("random_label, paired_label, paired_covariate_plus_label fall
back to ≈α or below without kind-level elevation"), `paired_label` ✓ and `paired_covariate_plus_label` ✓
but **`random_label` ✗** → **not yet a freeze-candidate**. The precise next problem (a P2.4c if authorized)
is the T-statistic's mild anti-conservatism on pure-noise labels — e.g. a finite-sample/df correction to
the cross-fit T, or a more conservative aggregation — *not* another null change. Still NO
freeze/confirmatory/real EEG.
