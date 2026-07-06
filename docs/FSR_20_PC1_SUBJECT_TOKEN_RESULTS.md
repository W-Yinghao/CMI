# FSR_20 — PC1-S Subject-Token Positive Control (results)

**Project FSR — Phase 4C.** Results of the pre-registered PC1-S injected positive control (FSR_19). CPU-only on the frozen 4B dumps + checkpoints; no GPU, no retrain, no target-label fit, α never chosen by target. Scripts + raw CSVs on branch `project/fsr-rq4-refit`; derived tables copied here (`results/fsr_pc1_subject_token/`). Seed 0, 21 folds.

## Headline
The FSR verification protocol **detects, localizes, and attributes** an injected harmful branch-local shortcut — but **erasure-based repair (even oracle) fails to recover it**, because the injected harm (like the natural spatial subject subspace, FSR_17) acts through **task-coupled** directions. Erasure is not a valid repair; a repair demonstration needs a *learned* shortcut (PC2, GPU) or a counterfactual/task-protected primitive (Phase 4D).

## Sanity (STOP-rules clear)
`α=0` reproduces the original logits/metrics exactly; the token produces a positive source class-directed logit shift (`token_class_shift_positive=True`); exact token subtraction recovers **1.0** — so the induced harm is 100% attributable to the injected token (no bug, no confound). Firewall: target labels used only for final scoring; token assignment, α selection, and repair fits are source-only.

## Status (terminology patch — PM Phase-4C review)
The original doc reported a single `detection_pass`. Per review this is **split** so the erasure-based
criterion (a *repair* test) is never folded into "detection":

```text
harm_induction_pass            = TRUE    (injected token induces target harm, CI excludes 0)
localization_pass              = TRUE    (harm localized to the injected spatial branch)
exact_attribution_pass         = TRUE    (exact token subtraction recovers ~1.0)
erasure_based_l5_pass          = FALSE   (erasing the injected token does NOT help target)
oracle_erasure_repair_pass     = FALSE
source_estimated_repair_pass   = FALSE
repair_pass                    = FALSE
--------------------------------------------------------------------
harm_localization_attribution_pass = TRUE   (= harm_induction ∧ localized ∧ exact_attribution)
```

**Claim language.** PC1 proves *"FSR detects, localizes, and exactly attributes a known harmful branch-local
shortcut; erasure fails as repair."* It does **not** prove *"FSR detects and repairs the shortcut."*

## Harm induction + localization + attribution — PASS
Injecting into `spatial_z` (the strongest natural candidate), induced target-bAcc harm rises monotonically with α and excludes zero at α≥1:

| α | induced harm (bAcc_orig − bAcc_inj) | 
|---|---|
| 0.0 | +0.000 |
| 0.25 | +0.004 |
| 0.5 | +0.014 |
| **1.0** | **+0.041 [+0.013, +0.072]** |
| **2.0** | **+0.066 [+0.028, +0.102]** |

**Localization** (α=1.0 induced harm): spatial **+0.041** > graph +0.021 > temporal +0.020 → the injected branch is correctly flagged as the most harmful. `harm_localization_attribution_pass=True` (harm + localized + exact-attributable).

## Erasure repair — FAILS (a finding, not a null)
Pooled recovery fraction at α=1.0 (`(bAcc_repaired − bAcc_injected)/(bAcc_orig − bAcc_injected)`):

| repair arm | recovery |
|---|---|
| R0 exact token subtraction (attribution upper bound) | **1.00** |
| R1 oracle token-**subspace** removal | **−0.19** |
| R2 source-estimated subject-subspace erasure | **−0.34** |
| R3 random-k control | −0.06 |

Every erasure arm makes the target **worse** (negative recovery), including the *oracle* that knows the injected token subspace. The `L5 "erasing helps"` criterion is also False at every α (task_drop stays ≥ 0). Mechanism: the injected harm acts through the class-directed (task-relevant) logit direction, so projecting out that subspace removes task-useful signal along with the token — exactly the task-entanglement the natural audit found in the spatial branch (FSR_17). Exact subtraction recovers fully only because it knows the precise per-sample vector (not a deployable repair).

## Verdict
```json
{"harm_localization_attribution_pass": true,
 "harm_induction_pass": true, "localization_pass": true, "exact_attribution_pass": true,
 "erasure_based_l5_pass": false, "repair_pass": false,
 "oracle_erasure_repair_pass": false, "source_estimated_repair_pass": false,
 "primary_branch": "spatial_z", "primary_alpha": 1.0,
 "alpha_selection_used_target": false, "target_labels_used_for_fit": false,
 "target_labels_used_for_final_eval_only": true}
```

## What this licenses
- **Proven:** the FSR protocol **detects and localizes** a known harmful branch-local shortcut on real EEG representations, with the harm exactly attributable to the injected direction. The verification chain (L1→L6 + localization + attribution) works as designed.
- **Proven (negative, and important):** **erasure is not a repair** — even an oracle that knows the injected token subspace cannot recover the target, because the harmful direction is task-coupled. This unifies with Phase 4B (natural spatial subject subspace is task-entangled) and TOS (`benefit_claimable=0/40`): removal hits task signal.
- **Not licensed:** any claim that FSR *repairs* a shortcut by erasure; any claim that natural source prevalence creates this shortcut (PC1 is a controlled injection).

## Recommendation (repair demonstration)
Because erasure fails even on a known injected shortcut, a *repair* contribution requires one of:
1. **PC2 prevalence-stress refit (GPU)** — induce `subject↔class` during *training* so the model *learns* a subject reliance; then subject-subspace erasure can undo the *learned* reliance (currently not approved — awaits PM go).
2. **Phase 4D counterfactual / task-protected repair** — a repair that neutralizes the subject-predictive component while *preserving* the task direction (e.g. counterfactual-consistency R4), rather than projecting out a task-coupled subspace.

Either way, PC1 has done its job: it validates the verification+localization+attribution machinery and shows, on a *known* shortcut, that erasure is the wrong repair primitive — which sharpens the Phase-4D repair design.

## Manuscript impact (Result 4, revised)
Result 4 becomes: *"An injected positive control confirms the framework detects and localizes a known harmful branch-local shortcut and attributes the harm exactly to the injected direction; but even oracle erasure fails to repair it, because the harmful direction is task-coupled — so repair must be counterfactual/task-protected, not erasure."* This is stronger and more honest than "erasure repairs," and it motivates Phase 4D (and, if approved, PC2).
