# Gate stress-suite — frozen design (P1.6; RUN in P2 after the selector is frozen)

Goal: decide whether the CMI domain-density gate is a **deployable harm-controller**, not just a strong score on
a diagnostic split. Built on real EEG features (erm + lpc dumps in `results/feat_dump*/`). Operates GPU-free.

## A0. MINIMAL FALSIFICATION SLICE (P2 step 1 — RUN BEFORE the full suite)
P1.4 disproves "more cohorts fixes PD" but does NOT establish that ONE CMI/density scalar solves both
sample-level concept-shift detection AND cohort-level adaptation harm. Before unifying them, falsify. Compare
exactly these scores, with a **prespecified orientation** and a **frozen aggregation**:
  1. current α-gate (entropy/uncertainty `g_unc`)
  2. `P(target|z)` domain-density
  3. covariate-shift-only (||μ_T−μ_S||, Bures)
  4. separability-only (target margin spread / class overlap)
  5. shift × separability (interaction)
  6. conditional/CMI score (residual decoder CMI / class-conditional density)
Against TWO targets: (a) sample-level `base-correct → adapted-wrong`; (b) cohort/batch-level mean loss increase.
**Unify into ONE gate ONLY IF** the same score wins both, same prespecified direction, frozen aggregation, on
**held-out cohorts AND held-out shift generators**. **Otherwise → a TWO-LEVEL controller**: batch/cohort-level
adaptation-eligibility (density/shift) + sample-level abstention (whatever wins sample-level). Do not hand-pick a
criterion after seeing the slice.

## A. Shift generators (real feature geometry; severity × prevalence curves, not one point)
Each generator G(strength θ, prevalence ρ) returns `(z', y', mask)` on a real target (z, y):
1. `clean` — no shift (false-abstention control).
2. `lowmargin_rot` — rotate boundary by θ; flipped samples are low-margin (MSP should catch).
3. `highmargin_cbw` — flip a high-confidence pocket + covariate signature (the dangerous case; MSP reverses).
4. `local_cond_rot` — class-conditional rotation in a localized feature region.
5. `prior_shift` — change target class prior (no concept change).
6. `covariate_shift` — affine covariate shift (alignment SHOULD help; gate should NOT abstain).
7. `random_noise` — flip ρ of labels at random (unstructured control; structured-vs-random separation).
8. `pure_conditional` — change P(y|z) with P(x) UNCHANGED — **NEGATIVE CONTROL.** DRY-RUN FIX: relabel a
   **margin-UNCORRELATED** subset (random over confidence), NOT a boundary rotation (which flips low-margin
   samples that MSP then catches trivially — a margin artifact, not conditional-shift detection). Verified: the
   density gate `domdisc` correctly fails (AUROC 0.52); MSP's 0.88 on the rotation variant was the margin artifact.
Sweep θ ∈ {0,10,20,30,45,60,90}°, ρ ∈ {0.05,0.10,0.15,0.20,0.30}.

## B. Prediction targets (per sample) — shift is NOT the goal; HARM is
- `shift_affected` = in the injection mask.
- `pred_error` = adapted prediction wrong.
- **`adapt_harm` = correct before adaptation, wrong after** (the primary target).
- `loss_increase` Δℓ_i = ℓ_i^adapted − ℓ_i^base.
Score every detector against EACH; **report `adapt_harm` AUROC/AUPRC as primary** (a gate that finds shifted
samples but not harmful ones is not deployable).

## C. Detectors / comparators (label-free on target)
confidence: `msp`, `entropy`, `energy`, `two_sided` (|conf−median|).
density/typicality: `mahalanobis` (to source class-conds), `knn_typicality`, `domain_posterior P(target|z)`,
`marginal_density_ratio`, `class_conditional_density`. **CMI gate** = domain-density (+ class-imbalance/separability
term, per the P1.4b harm hypothesis). Goal: isolate whether CMI's edge is *conditional structure* vs generic
*target typicality*.

## D. Gate-usage modes (compare at equal coverage)
- `post_abstain` — adapt, then reject high-risk outputs (caps output risk only).
- `pre_screen` — filter/down-weight unsafe samples BEFORE computing target alignment statistics (makes alignment
  itself safer; the P1.4b hypothesis — keeps ds002778's +7.5, drops ds004584's −0.5).
- `screen+abstain` — both.
Gate score computed on **pre-alignment** vs **post-alignment** representation (does post-alignment geometry mask
the danger? if so, fix the gate to pre-alignment space).

## E. Adapter-on-top matrix (gate must be adapter-independent)
{ERM, SPDIM, CITA, T3A} × {no-gate, CMI-gate}. The gate is feature-based ⟹ its harm-AUROC should be ≈constant
across adapters (orthogonal safety layer).

## F. Deployment/batch stress (NATURAL batches; no target labels used to form batches)
B ∈ {1,4,8,16,32,64}; class-imbalanced; near-single-class; batch-order permutation; independent vs
cumulative/streaming; threshold frozen on dev cohorts then transferred. Single-class / B<2 / severe-OOD / missing
channels ⟹ forced identity fallback.

## G. Metrics (all from ONE immutable predictions file)
prespecified-orientation AUROC; **adapt_harm AUROC + AUPRC (primary)**; risk–coverage; selective risk @ fixed
coverage; clean false-abstention rate; harm enrichment in the abstained subset; batch fallback rate; coverage
variability across cohorts; net avoided-harm − missed-benefit.

## H. Output schema (immutable; all tables/plots derive from it)
```
gate_stress_results.parquet  columns:
  cohort, seed, generator, theta, rho, adapter, gate_mode, gate_space(pre/post), detector,
  sample_id, base_pred, adapted_pred, y_true, gate_score, abstained,
  shift_affected, pred_error, adapt_harm, loss_increase
gate_stress_summary.md       — per (generator,detector,adapter,mode): harm-AUROC/AUPRC, risk@cov, avoided-harm
run_manifest.json            — code commit, feature-dump hashes, frozen selector, seeds, generator params
```

## Decision rule
The gate is a deployable controller IFF, across cohorts/seeds: (1) it predicts `adapt_harm` (not just
`shift_affected`) above the density/typicality comparators; (2) `pre_screen` ≥ `post_abstain` at equal coverage;
(3) harm-AUROC ≈ constant across adapters; (4) it fails as expected on `pure_conditional` (no false confidence);
(5) it holds on natural (non-diagnostic) batches with a dev-frozen threshold. Otherwise it is reported as a
diagnostic score, not a deployment mechanism.
