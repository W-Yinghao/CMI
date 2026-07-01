# CSC Route B3 — FREEZE-CANDIDATE PROTOCOL DRAFT (pre-registration; NOT yet frozen)

Status: **DRAFT for reviewer authorization.** This document specifies what a B3 confirmatory freeze WOULD
lock and how it WOULD be judged. It does **not** freeze anything, create a tag, or run a confirmatory. The
frozen A tag `csc-confirmatory-v1` / `dee8958` is untouched and unrelated. Real EEG is out of scope until a
synthetic confirmatory passes.

## Why now (evidence that warrants a freeze-candidate)

B3 reached a configuration whose **control type-I is stable across independent seeds under a locked rule**,
adversarially verified:
- **Controls ≤ α on two independent fresh seed blocks** (3000/4000 and 700000), by-kind × budget point AND
  CP-upper, pooled ≈ 0.004–0.005 (worst kind CP-up 0.036); `missing_pair`/`unequal_epochs_extreme` = 0/576;
  0 sampler failures; no hard-flags. The pre-lock P2.4c seeds (1000/2000) had `clean`@m30 = 0.052 > α, so
  the ≤α behaviour is attributable to the **rule**, not a lucky seed (red-team confirmed, no correction).
- **Primary power** `paired_concept` / `paired_concept_plus_cov` = **1.00** at m=20/30 in the four "strong"
  stress scenarios (baseline, high_nuisance, high_subject_tau, imbalanced).

This is DEVELOPMENT evidence; a freeze exists precisely to test it on **unseen** clusters.

## What would be FROZEN (method lock — no changes permitted after freeze)

```
method                = pc_centered_calibrated
h1_basis              = pc            (low-rank, rank=3)   condition_coding = centered (+-0.5)
regularisation        = C = 0.5
null                  = condition-matched FIXED-MARGIN h0 bootstrap (per-condition class margins preserved)
finite-sample gate    = studentized subject-consistency (Z = mean(delta_s)/se(delta_s))
guards                = pair_integrity >= 0.95 ; eligible complete pair needs >= 8 epochs/condition ;
                        min_confirm_pairs = 20 ; per-condition class coverage ; null invalid-accounting
decision (per budget) = fixed_margin mean-T p <= alpha_budget AND studentized p <= alpha_budget
                        AND LCB_{1-alpha_budget}(delta_s) > 0
alpha_family = 0.05 ; positive_decision_budgets = {20, 30} ; alpha_budget = 0.025 ; LCB level = 0.975
n_boot = 200 ; n_folds = 3
calibration_version   = p24d_cross_budget_alpha_spending_studentized_fixed_margin
```
A frozen manifest would hash the certifier source (`csc/mininfo/paired_calibrated.py`) + this parameter
block + the seed spec; the confirmatory runner would refuse to run if the hash or HEAD tag mismatches
(mirroring the A-line `frozen_code_provenance` discipline).

## The CLAIM (scoped, development-informed, honest)

- **Primary:** on paired within-subject ON/OFF targets with sound pairing (integrity ≥ 0.95, ≥ 8
  epochs/condition) and ≥ 20 eligible paired subjects labelled, the certifier outputs `CONCEPT_CONFIRMED`
  for genuine class-conditional / boundary concept shift (`paired_concept`, `paired_concept_plus_cov`)
  while controlling false confirmation at α on no-concept controls — **within the pre-registered operating
  envelope** below.
- **Pre-registered operating envelope (primary):** scenarios baseline / high_nuisance / high_subject_tau /
  imbalanced; budgets m ∈ {20, 30}. **Explicitly OUT of the primary envelope** (declared, not hidden):
  heavy label noise (~10% flips → primary power ~0.5, seed-variable) and very-short records (< 8
  epochs/condition → `NEED_MORE_LABELS` eligibility abstention).
- **Secondary (known-weak):** invisible pure-conditional relabel (`paired_pure_conditional`) — power ~0.25
  @ m=30; reported, NOT part of the primary confirmatory claim.
- **Out of scope:** real EEG / PD ON-OFF (only after a passing synthetic confirmatory), any Z-only
  (label-free) confirmation (theory: must abstain).

## Pre-registered CONFIRMATORY design (to run only after authorization)

The exact, machine-readable pre-registration lives in `csc/mininfo/b3_confirmatory_manifest.json` (method
lock + scenario_configs + seed spec + criteria + code hashes); the DRY-RUN validator
`csc/mininfo/run_b3_confirmatory.py` verifies provenance/scenario/seed and prints this plan, and refuses
`--execute`. Tests: `csc/tests/test_b3_confirmatory.py`.

```
seed block            = a SINGLE new unseen block, fixed in the manifest, DISJOINT from A's confirmatory
                        block (900000) AND every B dev seed {0.., 1000, 2000, 3000, 4000, 700000} AND all
                        smoke/test seeds (<100000)  =>  confirmatory base_seed = 1200000
clusters/cell         = 48 (controls) ; 48 (primary positives)
controls              = clean, paired_covariate, paired_label, random_label,
                        paired_covariate_plus_label, missing_pair, unequal_epochs_extreme
control scenarios     = ALL 6 stress scenarios (incl. label_noise AND few_epochs) -- controls are
                        evaluated everywhere, even in the 2 scenarios excluded from the primary POWER claim
primary positives     = paired_concept, paired_concept_plus_cov   (4 strong scenarios only)
secondary (reported)  = paired_pure_conditional (all scenarios; NOT gating)
decision              = ONE pre-registered CONJUNCTION verdict (no best-of-seed, no re-run)
runner                = fail-closed, frozen-code-hash guard, provenance-verified (A-line pattern)
```

### PASS criteria (CONJUNCTION — all must hold on the unseen block)

```
C1  missing_pair = 0 AND unequal_epochs_extreme = 0 false confirmations (all budgets; guards hold);
C2  every control kind x budget (pooled over all 6 scenarios, n=288): Clopper-Pearson one-sided upper
    (0.95) <= 0.05. NOTE: these are POINTWISE CP gates at the pre-declared reporting unit (kind x budget);
    the headline is a CONJUNCTION of pointwise gates, NOT a simultaneous familywise CI over all cells;
C3  no control cell (scenario x kind x budget, n=48) >= 6/48; AND no kind x budget has MULTIPLE (>=2)
    scenario cells at >= 3/48 (kind-level leakage). (Additional safety gate on top of C2.)
C4  primary positives: for EACH kind in {paired_concept, paired_concept_plus_cov} AND EACH budget
    m in {20, 30}, pooled over the 4 primary scenarios: Clopper-Pearson one-sided lower (0.95) >= 0.60;
    ADDITIONALLY no primary scenario x kind x budget cell may have point power < 0.50;
C5  0 sampler failures / null-invalid within cap ; all states in the 5-state set (no silent failure);
C6  independent red-team re-aggregation reproduces the verdict without correction.
```
Any single failure → the confirmatory FAILS (honest negative, like the A line). A pass is a
DEVELOPMENT-to-confirmatory success on **synthetic** data only.

## Honest limitations (must ship with any positive claim)

1. The operating envelope is **development-informed** (chosen from dev maps), so the confirmatory tests the
   scoped claim, not an unrestricted one; heavy-label-noise + short-record regimes are excluded by
   pre-registration, not by discovery-after-the-fact.
2. Primary power is 1.00 only in the 4 strong scenarios; label-noise power is weak and seed-variable, and is
   OUT of the primary claim.
3. `pure_conditional` (the subtlest shift) remains weak (~0.25) even with all machinery — a real limitation.
4. Everything is simulator-based; no real-EEG claim until a synthetic confirmatory passes AND a separate
   real-data authorization is given.

## What is NOT authorized by this draft

Creating the tag `csc-b3-confirmatory-v1`, writing the frozen manifest/hash, or running the confirmatory —
all require explicit reviewer authorization. Real EEG / PD ON-OFF remains blocked. This is a pre-registration
draft only.
