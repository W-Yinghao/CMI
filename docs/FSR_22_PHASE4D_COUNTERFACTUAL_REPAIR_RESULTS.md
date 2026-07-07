# FSR_22 — Phase 4D: Counterfactual / Task-Protected Repair (results)

**Project FSR — Phase 4D.** Results of the pre-registered Phase-4D repair (FSR_21). CPU-only on the frozen 4B
dumps + checkpoints; `PYTHONHASHSEED=0`, single-threaded; no GPU, no retrain, no target-label fit; α is the
pre-registered constant 1.0. Full 21-fold × {α=1.0, 2.0} run (SLURM CPU job 887402). Scripts + raw CSVs on
branch `project/fsr-rq4-refit`; derived tables copied here. Verdict independently recomputed + firewall-audited
(verification `wn9vpvc30`: all RNG-free numbers reproduce exactly, firewall clean).

## Headline — `repair_claim_level = none`, doubly supported
At the **pre-registered primary (α=1.0, pooled over all 21 folds)** the verdict is `none` for **two
independent** pre-registered reasons:
1. **Establish-harm gate fails.** Under the reproducibility-pinned seed the α=1.0 injection did not reliably
   induce harm: pooled induced harm **+0.018 [−0.025, +0.060]** (CI includes 0; 7/21 folds the injection did
   not harm). With no harm to repair on a third of folds, the repair endpoint is not evaluable.
2. **D1 does not beat the random control at the primary — it slightly loses.** Pooled recovery D1 **0.359** vs
   random-perturbation adapter D3a **0.404**; D1 − D3a pooled bAcc **−0.0008 [−0.025, +0.024]** (D1 beats D3a in
   only **7/21** folds). "D1 ≤ D3a" is itself a standalone `none` trigger (FSR_21), so the null holds even
   ignoring the harm gate.

A **secondary, exploratory** signal exists **only at α=2.0** and only as a **marginal pooled-mean** effect (see
panel); it is **not** the pre-registered headline and does **not** support a repair claim. Erasure (D2/D3b)
does not repair (confirms PC1).

## Verdict (pre-registered)
```json
{"repair_claim_level": "none", "counterfactual_repair_pass": false,
 "primary_alpha": 1.0, "harm_established_alpha1": false,
 "injection_harm_denominator_alpha1": 0.0179, "injection_harm_denominator_alpha1_ci": [-0.0248, 0.0599],
 "recovery_fraction_alpha1": 0.359, "randk_erasure_recovery_alpha1": 0.342,
 "d1_repaired_bacc_alpha1": 0.484, "d3a_repaired_bacc_alpha1": 0.485,
 "d1_minus_random_bacc_alpha1": -0.0008, "d1_minus_random_bacc_alpha1_ci": [-0.0252, 0.0236],
 "random_control_beaten_point": false, "source_val_task_safe": true,
 "exact_subtraction_recovery_alpha1": 1.0, "erasure_arms_excluded_from_headline": true}
```

## Primary panel (α=1.0, pooled all 21 folds — the pre-registered headline)
| quantity | value |
|---|---|
| pooled induced harm (bAcc_orig − bAcc_injected) | **+0.018 [−0.025, +0.060]** → harm **not established** |
| folds with harm > 0 / ≥ 0.02 | 14/21 / 9/21 |
| D1 (counterfactual adapter) pooled recovery / repaired bAcc | 0.359 / 0.484 |
| **D3a (random-perturbation adapter) pooled recovery / repaired bAcc** | **0.404 / 0.485** ← random ≥ D1 |
| **D1 − D3a (pooled bAcc)** | **−0.0008 [−0.025, +0.024]**; D1 wins 7/21 folds → **D1 does not beat random** |
| D2 task-orth erasure / D3b random-k erasure (recovery) | 0.199 / 0.342 (neither beats the random adapter) |
| D0 exact subtraction (attribution bound) | recovery **1.00** |

Reading: harm is absent/weak on a third of folds, and where recovery is defined the task-protected adapter is
**indistinguishable from — indeed marginally below — a random-perturbation adapter**. The "task-protected"
structure buys nothing over generic smoothing at the pre-registered strength.

## Secondary / exploratory panel (NOT the headline — α=2.0 only, marginal)
Only at the **secondary α=2.0** does a small separation appear, and only in the **pooled mean**:

| condition | n | D1 rec | D3a rec | **D1 − D3a bAcc [95% CI]** | folds D1>D3a | CI excl. 0? |
|---|---|---|---|---|---|---|
| α=2.0, pooled all folds (harm +0.062, established) | 21 | 0.689 | 0.258 | **+0.0265 [+0.0005, +0.057]** | 12/21 | **barely yes** |
| α=1.0, harm ≥ 0.02 *(target-harm-conditioned — diagnostic)* | 9 | 0.633 | 0.313 | **+0.0302 [−0.0002, +0.068]** | 5/9 | **no** |
| α=2.0, harm ≥ 0.02 *(target-harm-conditioned — diagnostic)* | 13 | 0.638 | 0.244 | **+0.0405 [+0.0035, +0.085]** | 8/13 | marginally yes |

Honest reading of the secondary:
- The D1-over-D3a separation is **α=2.0-only and marginal**: the α=2.0 all-folds CI **barely** excludes 0
  (lower bound +0.0005), and at the pre-registered **α=1.0 it does not** — even on the target-harm-conditioned
  harm≥0.02 subset the CI **includes 0** ([−0.0002, +0.068]).
- It is a **pooled-mean** effect, **not** per-fold uniform: D1 beats D3a in only 12/21 folds at α=2.0, and the
  positive margin is carried by ~3 high-harm BNCI2015 folds. "Consistent / wherever harm exists" would be
  false.
- The harm≥0.02 subsets are **target-harm-conditioned fold selections** (forbidden as a headline by our own
  STOP rule 4); shown only to characterize where the (weak) signal sits. Recoveries use ratio-of-pooled-means
  (FSR_21).
- The control is **fair, not degenerate**: D3a is a working adapter (positive consistency gain, ~0 clean-task
  drop, self-recovers 0.24–0.40 of harm), so D1's tiny α=2.0 edge is over a real baseline. But that edge is
  **bounded to the injected regime** — D1's augmentation family (`u_rand + v_c`) lives in the *same*
  class-directed subspace as the known injected token, so it partly benefits from knowing the token direction
  and need not transfer to a natural shortcut of unknown direction.
- The u-generalization diagnostic is healthy (source-val injected-task bAcc **0.678 [0.644, 0.710]**) — the
  adapter *does* generalize invariance to held-out source subjects, so the primary null is **not** merely a
  token-shift artifact; the adapter simply does not out-repair random at the primary.

## Why the primary is under-powered (honest disclosure)
Two compounding causes:
1. **Hashseed sensitivity.** PC1 (FSR_20, unpinned seed) had α=1.0 harm **+0.041** [established]. Phase 4D pins
   `PYTHONHASHSEED=0` (a design-red-team reproducibility fix) which re-draws the per-subject spurious class
   `c_target`; at low strength the induced harm is sensitive to whether that arbitrary class aligns with a
   target subject's true label. Under hashseed=0 the α=1.0 harm is **+0.018** (not established) while α=2.0 is
   **+0.062** (established, matches PC1's +0.066) — the reproducibility fix inadvertently under-powered the
   **α=1.0 primary**.
2. **A single anti-harm outlier fold.** On BNCI2015 subj-3 the injection *increased* target bAcc by **+0.29**
   (injected 0.828 vs orig 0.535) — the "shortcut" was task-*helpful* there, so it is not behaving as a
   shortcut. Dropping that one fold raises pooled α=1.0 harm to **+0.034** (above the 0.02 floor). That a single
   fold flips the floor **strengthens** the establish-harm gate's refusal to score recovery on this seed.

## What this licenses / does not
- **Does not license** (per pre-registration): any "counterfactual repair works" claim. The primary is `none`
  on two independent grounds (harm not established **and** D1 ≤ D3a). No headline claim.
- **Weak, transparent, non-headline:** at α=2.0 (secondary) a **marginal pooled-mean** separation of the
  task-protected adapter from a *functioning* random control exists (CI barely > 0), carried by a few folds —
  suggestive that task-protected structure *may* carry repair signal in the injected regime, but not
  established and not at the pre-registered strength.
- **Confirms:** erasure — even task-orthogonalized — is not a repair (PC1 + Phase 4D agree; D2/D3b do not beat
  the random adapter); exact subtraction (D0 = 1.0) still bounds attributability.
- **Firewall:** clean — target labels read only via `TargetScorer.score()` (per fold: 1 for bAcc_orig +
  6/α = 7 cumulative after α=1.0, 13 after α=2.0; no data-dependent extra reads, no feedback into fit/selection),
  `PYTHONHASHSEED=0`, α a pre-registered constant, source-only fit/selection, source-val holds out subjects.

## Recommendation
By the pre-registration, `repair_claim_level = none` at the primary → by the PM's rule **PC2 GPU stays
paused**. The honest recommendation is:

**Accept `none` and keep PC2 paused.** Phase 4D result: *verification succeeds; erasure is not a repair; a first
task-protected adapter did not clear the pre-registered primary and does not beat a random-perturbation control
there, with only a marginal α=2.0 pooled-mean hint.* Iterate the repair primitive (e.g. stronger consistency
weighting, deployment-matched multi-token augmentation, a repair that does not presuppose the token direction)
before any PC2.

> **What we must NOT do (STOP-8/STOP-9).** Re-freezing the Phase-4D primary at α=2.0 "because its harm is
> established," or raising the α=1.0 `scale` until harm ≥ 0.02, would be **post-hoc α/strength selection on
> already-seen folds** — precisely the maneuver the pre-registration forbids (α chosen by observed harm, which
> co-moves with the D1 benefit; and α=2.0 promoted to headline). If a retest at established harm is wanted, it
> must be a **strictly new pre-registration on a FRESH injection seed with the harm-target strength fixed a
> priori**, evaluated once — not a re-scoring of this verdict.

## Manuscript impact (Result 4, revised)
Result 4 stands as: *"Erasure — even oracle/task-orthogonalized — does not repair the injected shortcut. A
first task-protected counterfactual adapter did not clear the pre-registered primary (under a reproducibility-
pinned seed that under-powered the low-strength injection) and does not beat a random-perturbation control
there; only a marginal, α=2.0-only, pooled-mean separation hints at task-protected repair signal in the
injected regime. Repair therefore remains unconfirmed, sharply separating verification from intervention."*
