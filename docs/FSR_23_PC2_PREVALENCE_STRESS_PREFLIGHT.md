# FSR_23 — PC2 Prevalence-Stress Preflight (design only; NO GPU run)

**Project FSR — PC2 preflight.** Design of the prevalence-stress GPU refit that would induce a **learned**
subject↔class reliance during training (as opposed to PC1's *injected* latent-space shortcut), so that a
repair primitive can be tested against a shortcut the model actually *learned*. **This document is design
only.** Per PM decision (Phase-4C review): **the GPU run is NOT approved** — it is gated on Phase 4D. It
answers the six PM questions and pre-registers the go/no-go.

> **GPU GO/NO-GO (PM).** PC2 GPU runs **only if** Phase 4D's D1 counterfactual/task-protected repair reaches
> **≥ partial (moderate) pass** on PC1 (FSR_22). If D1 is `none` (fail), PC2 stays paused — fix the repair
> primitive first; do **not** run PC2 hoping erasure or a weak head will succeed on a learned shortcut.

## Why PC2 (and why after Phase 4D, not before)
PC1 injected a shortcut directly into a frozen latent — it proves detection/localization/attribution but
cannot test whether a repair fixes a *learned* reliance, because nothing was trained. PC2 makes the model
**learn** to rely on subject identity by stressing the training prevalence, then asks whether the Phase-4D
repair removes that learned reliance and recovers target accuracy. Running PC2 before a working repair
primitive risks a second erasure-style failure (PC1 already showed erasure is the wrong primitive), so the
order is fixed: **Phase 4D first → PC2 only if D1 works.**

## Constraints inherited (unchanged)
Refit scope = **ERM only**: no CMI loss, no fbdualpc, no architecture search, **no hyperparameter sweep**, no
target-label fit. The *only* deliberate change vs the F0 training run is the **source training prevalence**
(the subject×class sampling), plus applying the frozen Phase-4D repair at eval. **Target-label firewall:**
target labels are used only for final scoring (balanced accuracy — no target subsampling/selection), never for
fit, model/epoch selection, repair fit, or stress design.

---

## Q1 — How to induce subject↔class reliance in source training
Per LOSO fold (source = N−1 subjects, target = held-out subject), reweight/subsample **source trials** so that
subject identity becomes spuriously predictive of class **beyond** the genuine task signal:

- Assign each source subject `d` a spurious class `c_d` (deterministic, source-only — e.g. hash, as in PC1).
- Skew each source subject's **conditional** class prior `P(y | subject=d)` toward `c_d` at a controlled
  **stress strength** `ρ ∈ {0.0 (control), 0.5, 0.8}` (fraction of subject `d`'s trials that are class `c_d`,
  via class-stratified subsampling of that subject's trials). `ρ=0.0` = the balanced control refit.
- **Hold the source class marginal `P(y)` fixed** across ρ (see Q3): the stress lives entirely in the
  subject×class **joint**, not the overall class balance — so any learned effect is subject-coupling, not class
  imbalance. Enforce by choosing per-subject skews whose mixture reproduces the original `P(y)` (assign
  complementary `c_d` across subjects so subject-conditional skews cancel in the marginal).
- Train FBCSP-LGG with the **unchanged F0 ERM recipe** on this reweighted source. An ERM learner can now lower
  source loss by keying on subject-identifying features → a learned subject shortcut.

`ρ` is a **stress-design** parameter (source-only); it is never chosen by target accuracy.

## Q2 — How to keep the target broken / balanced
The held-out target subject is **not** part of the source skew, so the learned `subject→c_d` mapping does not
transfer: a model that over-relied on the subject shortcut wastes capacity and **mispredicts on target** — the
induced harm. Target trials are used **as-is**; class imbalance is handled by **balanced accuracy** (no
target-label-based subsampling to "balance" the test set, which would touch target labels for selection).
"Broken" = the spurious subject↔class structure is absent for the unseen subject by construction of LOSO.

## Q3 — How to confirm the model truly LEARNED subject reliance (not class imbalance / not nothing)
Three pre-registered checks, all reusing the Phase-4B branch-local L1–L5 machinery on the **refit** models:

1. **Reliance vs control (primary).** Stressed (`ρ>0`) vs control (`ρ=0`) refits, **matched source `P(y)`**:
   the stressed model must show (a) **higher subject-decodability** (L1 probe on its representation) **and**
   (b) **larger target harm attributable to the subject-predictive component** (L5: target bAcc rises when the
   subject reliance is removed by the Phase-4D repair — the reliance is *functional*, not just measurable).
2. **Class-imbalance confound ruled out.** Because `P(y)` is matched across ρ (Q1), a control refit with the
   *same* class marginal but **shuffled** subject↔class assignment (spurious structure destroyed) must **not**
   show the reliance — isolating subject-coupling from imbalance. This "shuffled-stress" arm is required.
3. **Dose-response.** Subject-decodability and target harm should increase monotonically with `ρ`
   (`0.0 < 0.5 < 0.8`); a flat curve ⇒ the stress did not induce learning ⇒ **STOP** (re-design, not a paper
   result).

## Q4 — Which repair primitive is tested
The **frozen winning Phase-4D primitive**:
- If Phase 4D D1 (counterfactual/task-protected repair) reached ≥ partial pass → **D1 is the PC2 repair**,
  re-fit source-only on the *stressed refit's* representation and applied at target eval (target labels score
  only). PC2 then asks: *does the task-protected repair remove a **learned** subject reliance and recover
  target bAcc, where erasure does not?*
- **D2 task-orthogonalized erasure** and **D3b random-k** run as **contrast baselines only** (claim-lock:
  erasure may never carry the repair headline, per FSR_21).
- If Phase 4D D1 failed → PC2 does not run (no primitive to test).

## Q5 — GPU budget
- Datasets: BNCI2014_001 (9 subj) + BNCI2015_001 (12 subj) → **21 LOSO folds**.
- Conditions: `ρ ∈ {0.0, 0.5, 0.8}` + `shuffled-stress` control = **4 source distributions**.
- Seeds: **3** (0,1,2) for stability (matches project multi-seed norm).
- Refits: 21 × 4 × 3 = **252 ERM refits**. FBCSP-LGG ERM ≈ a few–15 min/fold on V100/A40 →
  **~25–60 GPU-hours** total, per-fold checkpointed, embarrassingly parallel across folds/seeds/conditions
  (process-level parallelism, intra-op threads pinned). Repair (D1) fit is CPU (as in Phase 4D), negligible.
- Partition: V100 / A40 (per compute policy; no `--time`, per-fold checkpointing). Dumps latents +
  checkpoints exactly like Phase 4B so all analysis is CPU-side afterward.

## Q6 — When NOT to run (pre-registered stops)
```text
1  Phase 4D D1 did not reach >= partial pass (no working repair primitive) -> PC2 PAUSED.
2  A CPU preflight on a synthetic/small proxy fails to show the stress induces ANY subject-decodability
   dose-response (Q3.3) -> re-design the stress, do not spend GPU.
3  Enforcing matched P(y) across rho is infeasible for a fold (too few trials) -> drop that fold, disclose;
   do not silently imbalance.
4  Any design that reaches for CMI/fbdualpc/architecture search/hyperparameter sweep/target-label fit
   (forbidden) -> STOP; PC2 is ERM-refit + frozen Phase-4D repair only.
5  Reliance shown but the class-imbalance confound (Q3.2 shuffled-stress) is NOT ruled out -> STOP; the
   "learned reliance" is not separable from imbalance.
```

## Outputs (when/if approved)
```
docs/FSR_24_PC2_PREVALENCE_STRESS_RESULTS.md
results/fsr_pc2_prevalence_stress/
  pc2_stress_manifest.csv            # per fold/rho/seed: source P(y) match, subject skew, train/val ids
  pc2_reliance_vs_control.csv        # L1 subject-decode + L5 target-harm, stressed vs control vs shuffled
  pc2_dose_response.csv              # subject-decode + target harm vs rho
  pc2_repair_recovery.csv            # D1 (+ D2/D3b contrast) recovery on the LEARNED shortcut
  pc2_verdict.json                   # learned_reliance_induced, imbalance_confound_ruled_out,
                                     #   d1_repairs_learned_shortcut, repair_claim_level, firewall flags
  pc2_target_label_firewall.json
```

## Framing (fixed)
PC2 would test repair of a **learned** (still controlled) reliance induced by a source prevalence stress — not
a natural shortcut (Phase 4B remains `NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT`). A PC2 pass would license *"a
task-protected repair removes a learned subject reliance and recovers target accuracy where erasure fails"* —
**not** a DG method, SOTA, or a natural-harm claim. **This preflight commits no GPU; the run awaits an explicit
PM go conditioned on Phase 4D D1 ≥ partial.**
