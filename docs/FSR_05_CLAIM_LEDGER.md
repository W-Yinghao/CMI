# FSR_05 — Claim Ledger

**Project FSR — Step 2B.** Paper-ready status for every candidate claim, with the exact allowed wording and the forbidden wording. Statuses are **derived from the RQ result JSONs** by `scripts/fsr/build_claim_readiness.py` (not hand-set) and cross-checked against the PM's pre-stated expectation — all 10 match. Machine table: `results/fsr_phase2b/claim_readiness_table.csv`.

Status vocabulary: `READY` (state as a finding) · `READY_WITH_CAVEAT` (state with the caveat attached) · `SUPPORT_ONLY` (context, not a headline) · `NOT_READY` · `FORBIDDEN`.

| id | status | claim |
|---|---|---|
| C1 | READY_WITH_CAVEAT | Measured leakage magnitude does not certify reliance. |
| C2 | READY_WITH_CAVEAT | Task-head alignment is closer to reliance than raw leakage in frozen CIGL R3. |
| C3 | READY | Subject signal is erasable. |
| C4 | READY | Erasure strength does not certify target benefit. |
| C5 | READY_WITH_CAVEAT | Random-k falsifies non-specific NLL movement. |
| C6 | READY | Spatial branch is load-bearing. |
| C7 | SUPERSEDED (Phase 4B) | Branch-local leakage/reliance was missing in frozen artifacts (now measured by the 4B refit). |
| C8 | READY | CMI-control remains closed. |
| C9 | SUPPORT_ONLY | TTA-Control is positive but non-CMI. |
| C10 | READY | FSR is an audit framework, not a new DG method. |
| C11 | READY | FSR detects, localizes, and attributes controlled injected harmful shortcuts. |
| C12 | READY_WITH_CAVEAT | E4 repairs controlled first-moment constant-offset shortcuts. |
| C13 | FORBIDDEN | E4 repairs general or natural shortcuts. |
| C14 | NOT_ESTABLISHED (Phase 4G none) | E4b repairs controlled second-moment shortcuts. |
| C15 | NOT_READY (GPU_PAUSED) | PC2 learned-reliance repair works. |
| C16 | READY | Natural branch-local subject leakage is not automatically harmful. |
| C17 | READY_WITH_CAVEAT | Repair scope is first-moment-specific (1st-moment repairable, 2nd-moment not). |

> C11–C15 are the **repair-line** claims (Phases 4C–4G + PC2), derived from `pc1_verdict.json`,
> `phase4d/4e/4f_verdict.json` (hand-set from the frozen verdicts, each adversarially verified). C1–C10 above are
> the audit-line claims from the Step-2B RQ JSONs.

---

### C1 — Measured leakage magnitude does not certify reliance. — **READY_WITH_CAVEAT**
- **Ladder:** L1 → L5. **Evidence:** CIGL (RQ1B, RQ1C).
- **Allowed:** "In the recomputable seed0 slice, graph leakage is *negatively* associated with functional reliance (ρ=−0.42, n=42); the frozen pooled association is also negative (ρ=−0.342)." Both contradict the old 'more leakage ⇒ more reliance' direction.
- **Caveat (mandatory):** the pooled n=126 leakage correlation is `FROZEN_NOT_RECOMPUTABLE` (seeds 1/2 per-fold graph_kl were pruned); state it as a frozen-verified support result, not a full recomputation.
- **Forbidden:** "we fully reproduced the pooled leakage correlation"; "high leakage means the model relies on it."

### C2 — Task-head alignment is closer to reliance than raw leakage. — **READY_WITH_CAVEAT**
- **Ladder:** L4 vs L1 → L5. **Evidence:** CIGL (RQ1A, RQ3).
- **Allowed:** "Task-head alignment is positively associated with reliance (ρ=+0.338, n=126) and is the correctly-signed predictor; the Spearman difference align−graph_kl excludes zero (+0.816, seed0), so alignment is more positively associated with reliance than raw leakage."
- **Caveat (mandatory):** dataset-heterogeneous (2a significant, 2015 ns); within-group partial betas are not individually significant; alignment is **not** a validated estimator.
- **Forbidden:** "align_k2 is a validated reliance estimator"; "alignment has larger effect magnitude than leakage" (graph_kl has larger |β|, wrong sign).

### C3 — Subject signal is erasable. — **READY**
- **Ladder:** L3. **Evidence:** TOS_LEACE / mean_scatter / INLP / RLACE.
- **Allowed:** "LEACE drives linear subject decodability to chance on both backbones; INLP drives it to chance."
- **Caveat:** a nonlinear MLP residual persists after linear erasure (erasable ≠ fully removed).
- **Forbidden:** "LEACE removes subject leakage" (nonlinear residual remains).

### C4 — Erasure strength does not certify target benefit. — **READY** (main claim)
- **Ladder:** L3 → L6. **Evidence:** the 5 TOS erasers (RQ2 + RQ2 sensitivity, Step 2C).
- **Allowed (main, robust):** "No eraser certifies a proven target benefit — `benefit_claimable = 0/40` (positive bAcc point **and** 95% CI lower > 0, no task-collapse, no binary-harm, not random-k-matched)." This is the robust basis and is independent of the eraser subset.
- **Supporting negative-association result — `NOT_ROBUST_DO_NOT_HEADLINE`.** The all-cells correlation `corr(E_subject_removed, target_bAcc)` is −0.42 [−0.68,−0.11], but Step 2C shows it is **not robust**: it flips to **+0.54 (excludes 0)** on the principled-eraser subset (LEACE/RLACE only) and is ns when INLP and/or random_k are dropped. It is driven by INLP over-erasure and the random-k anchor, not a real "more removal → worse target" law. **Do not report the negative correlation as a finding.**
- **Forbidden:** "erasing subject signal improves DG"; "more subject removal harms the target" (the negative correlation is a confound artifact); counting a positive point estimate whose CI includes 0 as a benefit.

### C5 — Random-k falsifies non-specific NLL movement. — **READY_WITH_CAVEAT**
- **Ladder:** L3 control → L6. **Evidence:** TOS_random_k vs TOS_LEACE (RQ2).
- **Allowed:** "On 2a-TSMNet, LEACE's NLL drop (−0.031) is matched by same-k random removal (−0.034) while the subject stays decodable (0.998) → the NLL movement is non-specific regularization, not domain removal."
- **Caveat (mandatory):** flagged non-specific in 1/8 LEACE-vs-random_k cells — the claim is scoped to the flagged (canonical 2a-TSMNet) cell, not universal.
- **Forbidden:** "LEACE improves target NLL" as a DG claim.

### C6 — Spatial branch is load-bearing. — **READY**
- **Ladder:** L4. **Evidence:** FBCSP_LGG_branch_ablation (RQ4 descriptive).
- **Allowed:** "The spatial branch is load-bearing (zero_spatial ablation drop +0.074 on 2a / +0.088 on 2015; gate weight 0.489 / 0.572, highest of the three branches)."
- **Forbidden:** "spatial leakage is harmful" (no per-branch leakage probe).

### C7 — Branch-local leakage/reliance was missing in frozen artifacts. — **SUPERSEDED (Phase 4B)**
- **Status:** was the correct description at the frozen-artifact stage (no per-branch dump, no checkpoint). **Superseded by the Phase-4B ERM refit** (FSR_16/17), which produced a direct real-EEG branch-local L1–L6 audit (per-branch subject decode L1, ablation load L4, subject-subspace-erase replay L5, target metrics L6). The current status of the natural branch-local question is **C16**.
- **Historical allowed (frozen stage only):** "at the frozen-artifact stage the per-branch instrument did not exist." **Do not** present RQ4 as still blocked in the current manuscript.

### C8 — CMI-control remains closed. — **READY**
- **Ladder:** frozen premise. **Evidence:** CIGL_70, CMI_SYNTHESIS_01, CITA_03.
- **Allowed:** "Source-only and target-unlabeled CMI *control* are closed; FSR builds on the audit/measurement, not a revived control objective."
- **Forbidden:** any CIGL/FCIGL/dCIGL/MetaCMI/CITA-CMI positive-method claim.

### C9 — TTA-Control is positive but non-CMI. — **SUPPORT_ONLY**
- **Ladder:** L6. **Evidence:** TTA_Control_non_CMI (CITA_02/03).
- **Allowed (context only):** "Target-unlabeled adaptation (TTA-Control) improves target +0.037…+0.093 in all four cells — a genuine but **non-CMI** positive."
- **Caveat (mandatory):** seed0 only (no seeds 1/2); must be walled off from CMI-control; it is support/context for the FSR thesis, not an FSR headline.
- **Forbidden:** presenting it as evidence CMI-control works; treating the +gain as a CMI effect.

### C10 — FSR is an audit framework, not a new DG method. — **READY**
- **Ladder:** L1–L6 relationships. **Evidence:** the whole ledger.
- **Allowed:** "The contribution is the measurement→reliance→control audit ladder and the two boundary findings — *measurable ≠ relied-upon* and *erasable ≠ beneficial* — not a new DG method."
- **Forbidden:** any SOTA / new-DG-method framing.

### C11 — FSR detects, localizes, and attributes controlled injected harmful shortcuts. — **READY**
- **Ladder:** L1–L6 on an injected positive control. **Evidence:** PC1-S (`pc1_verdict.json`, FSR_19/20).
- **Allowed:** "On an injected subject-token positive control, FSR detects the induced target harm (+0.041/+0.066 bAcc at α=1/2), localizes it to the injected spatial branch (0.041 > 0.021 graph > 0.020 temporal), and attributes it exactly (token subtraction recovers 1.0); `harm_localization_attribution_pass = true`."
- **Caveat:** injected positive control, not a natural shortcut (4B natural = `NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT`).
- **Forbidden:** "FSR found a natural harmful shortcut"; framing PC1 as evidence natural prevalence creates the shortcut.

### C12 — E4 repairs controlled first-moment constant-offset shortcuts. — **READY_WITH_CAVEAT**
- **Ladder:** L6 repair on an injected first-moment positive control. **Evidence:** Phase 4F (`phase4f_verdict.json`, FSR_26/27); Phase 4D/4E `none` (erasure + counterfactual head fail).
- **Allowed:** "E4 first-moment mean alignment provides a target-X-only repair for a controlled first-moment constant-offset injection, with a small shortcut-specific gain (absolute +0.033 bAcc, clustered CI [0.009, 0.058]) after netting the generic TTA benefit and beating a random-direction control." `repair_claim_level = strong`, qualified `strong_within_controlled_first_moment_scope`.
- **Caveat (mandatory, all):** 73% of the recovery is a **mechanical identity** (first-moment aligner inverts a first-moment offset by algebra; non-identity netted 0.68); it **fails leave-one-dataset-out** (BNCI2014 harm/specificity not established — BNCI2015-carried); the pre-registered LOSO-seed axis is near-inert; E4 **overshoots** orig and helps clean models → generic domain-mean TTA, not surgical removal.
- **Forbidden:** "E4 is a general shortcut repair method"; "E4 surgically removes shortcut information"; "E4 repairs natural EEG subject shortcuts"; leading with the 0.93 ratio.

### C13 — E4 repairs general or natural shortcuts. — **FORBIDDEN**
- **Why:** E4's scope is a construction-matched first-moment positive control that fails leave-one-dataset-out; 4B/4D natural = `none`. No evidence licenses general/natural repair.
- **Forbidden:** any statement that E4 (or FSR) repairs natural, learned, or general shortcuts / harmful subject leakage; any DG/SOTA framing.

### C14 — E4b repairs controlled second-moment shortcuts. — **NOT_ESTABLISHED (Phase 4G = none)**
- **Evidence:** Phase 4G (`phase4g_verdict.json`, FSR_29/30). On a strictly mean-null second-moment injection, covariance-shrinkage (E4b) does **not** beat random-direction shrinkage at the source-selected operating point (`E4b − E3 = +0.005 bAcc`, clustered CI [−0.005, 0.014], < 0.02); **even oracle** shrinkage of the true `v_c` is sub-DELTA (`fail_attribution = genuinely_weak_second_moment_repair`). Repair is **first-moment-specific**.
- **Allowed (negative, scoped):** "A controlled mean-null second-moment stochastic shortcut is not repaired by covariance-shrinkage at the source-selected operating point, even with oracle knowledge of the injected direction; a DELTA-clearing direction-specific advantage appears only in the injection-dominant (α=3) near-tautological regime. The deployable repair family (4F+4G) is confined to first-moment deterministic offsets."
- **Forbidden:** "E4b repairs second-moment/covariance shortcuts"; "second-moment shortcuts are unconditionally unrepairable" (α=3 injection-dominant does cross the bar); any learned/natural repair claim.

### C15 — PC2 learned-reliance repair works. — **NOT_READY (GPU_PAUSED)**
- **Status:** PC2 GPU is **paused** (`pc2_gpu_run_authorized = false`); the PC2-E4 readiness preflight (FSR_28/31) is design-only. Learned reliance ≠ clean first/second-moment offset; guarded/possibly-negative expectation. Two blockers (FSR_31): only 2 preset-ready datasets (< the ≥3 the binding leave-one-dataset-out gate needs) and Phase 4G = none.
- **Forbidden:** any statement that PC2 / learned-reliance repair works or is being run; presenting PC2 in results (future-work/preflight only).

### C16 — Natural branch-local subject leakage is not automatically harmful. — **READY**
- **Ladder:** L1→L6 direct real-EEG branch-local audit. **Evidence:** Phase 4B ERM refit (FSR_16/17; `NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT`).
- **Allowed:** "In the Phase-4B branch-local audit the spatial branch is the strongest subject-leakage and load-bearing candidate, but subject-subspace removal *hurts* target performance (task_drop positive, +0.050* on BNCI2015); FSR therefore refuses to call it a harmful shortcut — it is a task-entangled / task-useful reliance."
- **Forbidden:** "spatial leakage is harmful"; "erase spatial subject directions to improve EEG DG"; "graph leakage is benign in general"; "there is no subject leakage" (it is high, ~0.92 decodable — the point is measurable ≠ harmful).

### C17 — Repair scope is first-moment-specific. — **READY_WITH_CAVEAT**
- **Ladder:** L6 repair on injected positive controls. **Evidence:** Phase 4F (`strong_within_controlled_first_moment_scope`) + Phase 4G (`none`).
- **Allowed:** "E4 (first-moment mean alignment) repairs a controlled first-moment constant-offset injection, while E4b (covariance-shrinkage) does not repair a controlled mean-null second-moment injection at the source-selected operating point (even oracle-directed); the deployable repair family is confined to first-moment deterministic offsets."
- **Caveat (mandatory):** the 4F first-moment pass is construction-matched (73% mechanical identity, BNCI2015-carried, fails leave-one-dataset-out); the 4G second-moment negative is at the source-selected operating point (a direction-specific advantage appears only in the injection-dominant α=3 near-tautology).
- **Forbidden:** "E4/E4b repairs general shortcuts"; "second-moment shortcuts are unconditionally unrepairable"; "FSR solves shortcut repair."

### C18 — Source prevalence-reweighting does not weaponize the natural subject signal (head-only). — **READY_WITH_CAVEAT**
- **Ladder:** L5 learned-reliance inducibility on a cheap linear head. **Evidence:** Phase 7B (`head_verdict.json`, FSR_39; `gate_pass=false`, fail-closed).
- **Allowed:** "Source subject-class prevalence-reweighting (on true labels) induces the subject↔class correlation *exactly* but does **not** induce a learned head-level reliance on the frozen 4B representation — the task signal is a sufficient statistic, so the head keeps the generalizing signal and preserves task accuracy; the learnability gate fails-closed. A label-corruption positive control collapses task bAcc ~5× (drop +0.179 at ρ=0.8), so the gate has power."
- **Caveat (mandatory):** this is **"no weaponization *demonstrated* under prevalence-reweighting,"** NOT "the head resists weaponization" (the learn-then-decline leg was never achieved); Kish eff-n falls to 0.735 at ρ=0.8.
- **Forbidden:** "the head resists weaponization"; "natural subject signal is safe / can never be harmful"; reporting the null as a robustness finding.

### C19 — A learnable subject-conditional task-conflict corruption does not weaponize into transferable structure-specific harm (head-only). — **READY_WITH_CAVEAT**
- **Ladder:** L5→L6 weaponization test on a cheap linear head, staged (Q7C-a learnability → Q7C-b transfer). **Evidence:** Phase 7C (`label_conflict_verdict.json`, FSR_40/41; `heldin_learnability_pass=true`, `pseudo_target_transferability_pass=false`, `weaponization_confirmed=false`); design-red-teamed + adversarially verified.
- **Allowed:** "A subject-keyed, task-conflicting label corruption (global P(y) held exact) is **learnable in-sample** by a linear head on frozen 4B latents (conflict-subset fit 0.70 vs a 0.20 clean-head floor, dose-monotone), but it does **not weaponize**: on held-out subjects and on the target its true-task harm does **not exceed** a subject-scrambled control (`beats_shuffle=false` on both datasets) and is not localized to the subject subspace (erasure ≤ 0); it beats only matched random noise (+0.078), so the transferable damage is **generic subject-blocked corruption harm, not a subject-structure-specific reliance**."
- **Caveat (mandatory):** Q7C-a is **in-sample fittability, NOT** "a subject *shortcut* was learned" (l5/ERASE diagnostics ≤ 0); the mildly-negative pooled vs-shuffle is significant only pooled / only on the binary dataset (BNCI2014_001 vs-shuffle CI crosses 0 — a null), and a negative structured-minus-shuffle is a **construction asymmetry**, not "scrambling is more harmful"; achieved corruption rate saturates at ~0.34 at nominal γ=0.4 (disclosed).
- **Forbidden:** "a subject shortcut was learned / weaponized"; "task-conflict corruption weaponizes subject leakage"; "scrambling is more harmful than the true structure"; "E4 repairs the weaponization" (none demonstrated — E4's +0.028 recovers *generic first-moment* harm only, 4F-consistent, `repair_claim_level=null`); any natural-harm/clinical/DG/SOTA framing.

---

## Wording discipline (applies to every write-up)

- Never state a leakage or erasure *measurement* as a reliance or benefit *conclusion* (C1/C4 are the guardrails).
- Always attach the mandatory caveat to C1, C2, C5, C9.
- The raw `improves_target` flag is `raw_improves_target_flag`; only `benefit_claimable=YES` (proven bAcc gain, no collapse, no harm, specific) licenses a benefit claim — and it is 0/40.
- Provenance tiers are load-bearing: `RECOMPUTED` (align n=126) ≠ `RECOMPUTED_SIGN_ONLY` (graph_kl seed0) ≠ `FROZEN_NOT_RECOMPUTABLE` (graph_kl pooled).
