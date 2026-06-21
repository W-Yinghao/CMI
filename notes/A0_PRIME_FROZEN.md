# A0′ — CONFIRMATORY gate slice (PRE-REGISTERED, frozen 2026-06-21, before running)

Purpose: CONFIRM, on genuinely **unseen generator realizations**, the exploratory A0-adversarial finding — does
**post-alignment boundary fragility (s_sep, g_unc) predict adaptation-SPECIFIC harm**, and are **CMI/support
anti-aligned with harm** (suited to detecting shifted geometry, NOT to gating harm)? Does NOT change project
positioning; only after A0′ reproduces on unseen realizations do we reconsider framing. Freeze B still needs a
later minimal closed-loop pilot (tentative-adapt vs batch-rollback vs sample-abstention). GPU-free.

## Tightenings over the exploratory pass (mandatory)
1. **Unit of replication = (batch, generator-family, cohort), NOT sample count.** All metrics macro-averaged over
   these units; per-sample N is never treated as the power source.
2. **Decompose** every post-alignment score: `s_i = s̄_B + (s_i − s̄_B)`. Test the **batch mean** `s̄_B` and the
   **within-batch residual** `(s_i − s̄_B)` SEPARATELY — pooled AUROC may only be separating high-risk batches.
3. **Post-alignment scores are ROLLBACK/ABSTENTION-after-tentative-adaptation**, NOT a pre-adaptation gate (they
   require computing the adaptation first). Labelled as such in the decision.

## Unseen realizations (else "confirmation" is just a new statistic on the same outputs)
A0 used DETERMINISTIC generators. A0′ uses **stochastic** generators drawing a fresh realization per seed, from a
seed family (`real ∈ {0..4}`, base offset 9000) NOT used before; R=5 realizations per (cohort, generator, severity).
Stochasticity: `lowmargin_rot` random boundary-orthogonal plane; `highmargin_cbw` random subset of the
high-confidence pocket + random covariate direction (component along w); `covariate_shift_beneficial` random
covariate direction. FROZEN (identical to A0): readout/adaptation (bit-exact, ev split, matched_coral, shrink=0.1),
B=32 recording-order batches, every score formula + direction, equivalence margin 0.03.

## PRIMARY SAMPLE TEST — within-batch sample abstention (post tentative adaptation)
- Target: `harm_flip = 1[base-correct → adapted-wrong]`, evaluated on the base-correct subset only.
- Score: **post-alignment `s_sep`**.  Comparator: **post-alignment `g_unc`**.
- Eval: rank by the **within-batch residual** `(s_i − s̄_B)`; **AUROC/AUPRC computed WITHIN each eligible batch**
  (≥8 base-correct, both flip classes present), then **macro-average batch → generator-family → cohort**. Report PD
  and SCZ **separately**. Sample inference clustered by (cohort × generator-family × batch).

## PRIMARY BATCH TEST — rollback eligibility
- Target: `mean Δℓ_B` (continuous).  Score: the score's **frozen mean-over-batch** `s̄_B`.
- Eval: Spearman + C-index + AUROC[`mean Δℓ_B > 0`]; **leave-one-generator-family-out**; disease-stratified.

## Mandatory recorded quantities (no pooled-only reporting)
- harmful-flip rate, beneficial-flip rate (adapted-correct ← base-wrong), and net (harmful − beneficial);
- **base-error AUROC vs adapted-error AUROC** per score — distinguishes "finds ordinary hard samples" from
  "finds adaptation-specific harm" (harm-flip AUROC must exceed base-error AUROC to be adaptation-specific);
- selective risk at **fixed 80% coverage** (within-batch);
- **global top-20% vs within-batch top-20%** reported SEPARATELY;
- **direction per generator family** — pooled-only direction is forbidden.

## Decision (FOUR outcomes; admissible = stable on the confirmation set, AUROC>0.5 + within 0.03 of best, consistent sign/direction across generator-families and PD/SCZ)
- `SINGLE_SCALAR_CANDIDATE` — the SAME score is admissible for BOTH the batch-mean (rollback) AND the within-batch
  residual (abstention) tests.
- `TWO_LEVEL_CANDIDATE` — different scores admissibly handle rollback-eligibility vs sample-abstention.
- `POST_ALIGN_ABSTENTION_ONLY` — only the within-batch sample ranking holds (rollback-eligibility does not).
- `DIAGNOSTIC_ONLY` — nothing reproduces stably on the confirmation set.

## Interpretation guard for CMI/support
AUROC < 0.5 for `s_support`/`cmi` here means **anti-aligned with adaptation HARM** — a DIFFERENT endpoint from the
boundary-shift *affectedness* they were validated on earlier. The candidate mechanism finding (to CONFIRM, not yet
assert): *density/CMI identify shifted geometry yet are anti-aligned with adaptation harm, while post-alignment
boundary fragility predicts harmful transitions.* This is recorded as a mechanism-separation hypothesis; project
re-direction waits on A0′ reproduction AND the later closed-loop pilot.

## Output (immutable)
```
results/a0_prime/<freeze_hash16>/
  a0prime_summary.json   # 4-way decision; per-disease within-batch & batch-mean tables; per-generator-family
                         # directions; base-vs-adapted-error AUROC; flips/net; selective-risk; global-vs-withinbatch
  run_manifest.json      # frozen readout verify, seed family (UNSEEN), dump hashes, this file's hash
```
