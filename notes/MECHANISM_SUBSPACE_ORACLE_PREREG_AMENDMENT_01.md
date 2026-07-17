# Mechanism-Subspace Oracle — PRE-REG AMENDMENT 01 (M0.1 revision). SPEC ONLY — M1 HOLD until PM review.

Amends `MECHANISM_SUBSPACE_ORACLE_PREREG.md` per PM review. Scientific exploration is ACTIVE; a negative
terminates only the current hypothesis/objective/implementation, never a direction — only the PM stops a line.
Manuscript FROZEN. No `CLOSE/CLOSED/closeout/CANCELLED/learn nothing` wording anywhere.

## P0.1 — result-routing replaces all close wording; every negative writes a failure record
M1 verdict is one of (graded; NONE is a stop):
`MECHANISM_ENRICHED_OVER_RANDOM | GENERIC_LOW_RANK_EFFECT | NO_DETECTED_MECHANISM_ENRICHMENT |
BASIS_UNSTABLE_OR_RANK_DEFECT | READOUT_DEPENDENT | BACKBONE_SPECIFIC | INCONCLUSIVE`. Any non-`ENRICHED`
verdict MUST emit `{failure_layer, evidence, learned_lesson, next_hypothesis, next_experiment}` — the negative
routes to a next experiment (see routing table), it does not end the subspace direction.

## P0.2 — utility naming: dU, not dI
All effects are query balanced-accuracy differences -> `dU_mechanism_specific`, `dU_source_meta`. CMI appears
ONLY in post-hoc certification, never as a selection target or in the effect name.

## P0.3 — PRIMARY = class-contrast disagreement (secondary demoted)
Candidate family order (frozen): 1) `B_contrast_disagreement` PRIMARY; 2) `B_rule_disagreement` secondary;
3) `B_gradient_disagreement` secondary; 4) `B_cond` NEGATIVE REFERENCE. In the source-whitened metric, for
subject d and class pair (a,b): `c_d^{ab} = μ_{d,a} − μ_{d,b}`; shared `c̄^{ab} = Σ_d w_d c_d^{ab}`; residual
`r_d^{ab} = c_d^{ab} − c̄^{ab}`. 4-class = all 6 unordered pairs; binary = the single contrast.

## P0.4 — B_rule = shared-plus-residual SHRINKAGE (not free per-subject heads)
`W_d = W_0 + ΔW_d`, minimize `Σ_d L_d(W_0+ΔW_d) + λ0‖W_0‖_F² + λΔ Σ_d‖ΔW_d‖_F²`; `G_rule = Σ_d ΔW_dᵀ ΔW_d`.
(λ0, λΔ) fixed source-only or by nested source-CV — NEVER seeing target utility. Separates shared rule /
subject residual / finite-sample noise (the free per-subject fit was the earlier estimator defect).

## P0.5 — B_grad = class-CONDITIONAL gradient disagreement (not the marginal head-row gradient)
`g_{d,y} = E[∇_z ℓ(h(z),y) | D=d, Y=y]`; `r_{d,y} = g_{d,y} − ḡ_y`; `G_grad = Σ_{d,y} w_{d,y} r_{d,y} r_{d,y}ᵀ`.
Class-BALANCED and subject-BALANCED; numerical-rank thresholded; save the raw singular spectrum and report the
theoretical rank bound vs head rowspace. `B_grad` is the current predictor ERROR geometry, not the full
representation mechanism, and is labelled as such.

## P0.6 — generalized eigenproblem for CONSTRUCTION; only source-safety is a HARD gate
`B_mech = top-ρ eigenvectors of G_dis v = ρ (G_shared + η I) v` (η small, frozen source-only). The ONLY hard
gate at selection is `ΔR_source ≤ 0.02`. The shared-mechanism preservation is a CONTINUOUS diagnostic +
sensitivity gate `s(P) = tr(P G_shared)/tr(G_shared)` (report it; do NOT hard-cut a second time — avoids a
second accidental empty feasible set). No hard `tr(P G_shared) ≤ ε`.

## P0.7 — two random controls (task-overlap-matched is the PRIMARY specificity control)
For each cell, ≥100 AMBIENT random dictionaries (match rank / action count / greedy budget / T_cal access) AND
≥100 SHARED-OVERLAP-MATCHED random dictionaries (additionally match `s(P) = tr(P_R G_shared)/tr(G_shared)` to
the informed dictionary), each in two independent 100-draw blocks (Monte-Carlo stability). Primary specificity
= informed − E[task-overlap-matched random]; ambient random = generic low-rank control.

## P0.8 — construction, selection, and evaluation are three separate steps
1) SOURCE-ONLY basis construction `B_mech = eig(G_dis, G_shared)` (no target). 2) NON-DEPLOYABLE existence
selection: within the `B_mech` action family choose subset/rank using `Y_cal` (hindsight, not deployable).
3) Query evaluation: score ONLY on `T_query` (future-session, session-macro). M1 tests whether the SOURCE-
derived mechanism dictionary enriches TARGET-beneficial actions over a task-overlap-matched random dictionary.

## P0.9 — matrix arithmetic
M1 = (9 + 12) subjects × 2 backbones × 3 seeds = **126 fold-seed cells** (NOT 2 datasets × 2 backbones × 21 ×
3 = 252). The dataset is implicit in the subject id. Missing cells explicit in a completeness matrix.

## M1 result-routing (no closeout; each result opens a next experiment)
| result | failure layer | next |
|--------|---------------|------|
| MECHANISM_ENRICHED_OVER_RANDOM | none | M2 source identifiability |
| GENERIC_LOW_RANK_EFFECT (gain>0 == matched random) | objective/geometry | revise mechanism dictionary or training-time regularizer |
| contrast enriched, rule/grad not | per-subject head/grad estimator noise | keep contrast, stop estimator expansion |
| rule/grad enriched, contrast not | mechanism at decision boundary not mean-contrast | design learned rule-disagreement projector |
| NO_DETECTED_MECHANISM_ENRICHMENT (all post-hoc) | ERM latent has no redundant deletable task pathway | pivot to TRAINING-TIME shaping `L = L_task + λ Σ_{d,a<b}[1 − cos(c_d^{ab}, c̄^{ab})]`, re-run the SAME oracle |
| BACKBONE_SPECIFIC | representation-capacity dependence | analyze latent rank / direction consistency / head geometry; confirm on new seeds or a 3rd backbone |
| unconstrained enriched but source-safe not | beneficial direction entangled with source task | TTE / representation reshaping, not post-hoc deletion |

## Gates + compute (unchanged from M0.1 base except the above)
M1 existence → M2 source identifiability (LCB95(dU_source_meta)>0 AND ≥25% recovery) → M3 learned oblique
`P=UUᵀ`, U⊆span(B_contrast,B_rule,B_grad), rank≤3, frozen encoder + fresh head → M4 TTE. CPU/high-memory for
M1 (no GPU for numpy eig); GPU only from M3 (learned projector / retraining), still full LOSO × 3 seeds × both
datasets × matched random × cluster CI, no favorable-subject selection. Synthetic/smoke = engineering only.
M1 run HOLD until this amendment + implementation tests + a real-EEG engineering smoke are reviewed.
