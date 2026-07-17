# Mechanism-Subspace Oracle — FROZEN pre-registration (M0.1). SPEC ONLY — M1 HOLD until PM review.

New method line after the conditional-subject / target-X G1 selector `FAILED_UNDER_CURRENT_OBJECTIVE` (the
informed B_cond oracle showed `NO_ENRICHMENT_OVER_RANDOM` on the fair equal-budget control; only the PM
declares a scientific line stopped). Branch `agent/cmi-trace-mechanism-subspace-oracle` (base = corrected
target-X head, CLOSED wording revoked). Manuscript FROZEN. CMI = measurement ruler only, never the objective.
No `CLOSED/closeout/CANCELLED` wording — negative results use graded stage verdicts + result-routing.

## What the failed line taught us (carried forward)
1. Subject-identity subspace (B_cond) is NOT the DG-failure subspace: B_cond answers "which directions separate
   subjects?"; DG needs "which TASK-bearing directions do source subjects implement DIFFERENTLY?".
2. Most safely-removable leakage is functionally UNUSED (lives in ker(W_c)): deleting it drops CMI + keeps
   softmax but cannot move DG. (Track B B0-v2: cond head-overlap enrichment ~1.0-1.2 = barely above isotropic.)
3. A hard `U ⊆ row(W_c)` anchor + source-safety is empty by construction (deleting task-used dirs hurts task).
4. Any candidate dictionary must first beat an EQUAL-BUDGET random-basis oracle (select-on-cal, score-query).

## Objective shift
FROM `min I(Z;D|Y)` TO: **delete the task-bearing directions where source subjects DISAGREE on the decision
mechanism while PRESERVING the shared mechanism.** Utility-first (Δ_query), CMI post-hoc certification only.

## PRIMARY object: class-contrast disagreement (source-only, whitened metric)
For each subject d and class pair (a,b): `c_d^{ab} = μ_{d,a} − μ_{d,b}` (whitened). 4-class = all 6 unordered
pairs; binary = the single contrast. Shared `c̄^{ab} = Σ_d w_d c_d^{ab}`; residual `r_d^{ab} = c_d^{ab} − c̄^{ab}`.
`G_shared = Σ_{a<b} c̄^{ab} c̄^{abᵀ}`, `G_dis = Σ_{d,a<b} r_d^{ab} r_d^{abᵀ}`. Solve the generalized eigenproblem
`G_dis v = ρ (G_shared + η I) v`; high-ρ directions = large cross-subject contrast disagreement with weak shared
contrast. `B_contrast` = top-ρ (numerical-rank thresholded). Rationale over free per-subject logistic heads:
fewer params, small-EEG-stable, well-defined for 4-class, source-subject bootstrap-able, connects to FMScope
within-subject direction consistency, no calibration dependence.

## SECONDARY objects (corrected)
- `B_rule` (multi-task RIDGE, not free per-subject): `W_d = W_0 + ΔW_d`, penalize `λ‖ΔW_d‖_F²`; disagreement =
  subspace of {ΔW_d}. Numerical-rank thresholded. (Free per-subject W_d was the earlier estimator defect.)
- `B_grad`: `g_d − ḡ`, numerical-rank thresholded (grad ∈ row(W) so true rank ≤ C−1), per-subject/class
  normalized, shared-head scaling removed; save singular spectrum.
- `B_rule ∪ B_grad`: SECONDARY union only (not a first-round mega-dictionary).
- `B_cond`: NEGATIVE REFERENCE (the failed family).

## Candidate families (M1): contrast_disagreement (PRIMARY), rule_disagreement, gradient_disagreement, B_cond (neg ref)
Each: source Ledoit-Wolf whitened metric; numerical-rank threshold; rank budget k ≤ 3; select on T_cal labels;
score on T_query future-session (session-macro); report BOTH source-safe and unconstrained oracle.

## Random controls (two kinds, ≥100 each, two independent 100-draw blocks for Monte-Carlo stability)
- AMBIENT random: same dictionary rank / candidate count / greedy + rank budget / calibration-label access.
- TASK-OVERLAP-MATCHED random: additionally match `tr(P_B P_shared)` (or task-head overlap) so candidate vs
  random are not confounded by task sensitivity.
Primary: `ΔU_mechanism_specific = U_mechanism_oracle − E_R[U_matched_random_oracle]` (subject-cluster CI). CMI
is measured post-hoc, never selected on.

## M1 result-routing (NO closeout wording; each result opens a specific next branch)
| result | condition | next |
|--------|-----------|------|
| A | LCB95(ΔU_mechanism_specific) > 0 on ≥1 family, no clear harm on other dataset | source-identifiability → target-X observability → learned oblique → TTE |
| B | absolute oracle gain > 0 but == matched random (INCONCLUSIVE / NO_ENRICHMENT) | revise candidate geometry (new disagreement object), not stop |
| C | all post-hoc oracles show no gain | pivot to TRAINING-TIME shaping: `L = L_task + λ Σ_{d,a<b}[1 − cos(c_d^{ab}, c̄^{ab})]` (or grad-disagreement penalty), re-run the SAME oracle audit on the reshaped representation |
| D | holds on ONE backbone only | analyze latent capacity / task-direction consistency / numerical rank / head geometry; confirm on new seeds or a 3rd backbone before any method claim |

## Staged gates (each FULL: LOSO × 3 seeds × BOTH datasets, subject/fold-cluster CI, equal-budget random)
M1 existence (above) → M2 source identifiability (source-LOSO meta selection; LCB95(ΔU_source_meta) > 0 AND
≥ 25% recovery of the M1 oracle gain) → M3 learned oblique `P = UUᵀ`, `U ⊆ span(B_contrast,B_rule,B_grad)`,
rank ≤ 3, frozen encoder + fresh head, vs equal-budget random + CMI certification → M4 TTE (only if a STATIC
learned subspace gives specific full-EEG target gain).

## Matrix + compute
M1 = 2 datasets × 2 backbones (EEGNet primary, DGCNN secondary/stored-head) × 21 subjects × 3 seeds = **126
fold-seed cells**; missing cells explicit in a completeness matrix. Oracle + linear-subspace math on CPU/high-
memory arrays. GPU only from M3. Synthetic/smoke = engineering only. No TSMNet/foundation/new backbones.

## Discipline
Pre-reg + red-team before, adversarial-verify after; utility-first; CMI post-hoc; source-safety hard. Allowed
non-negative outcome: mechanism-alignment utility that is NOT CMI-specific -> a general transductive-alignment
finding, not a CMI/TOS method. No manuscript. M0.1 is SPEC ONLY; M1 HOLD until PM reviews this spec + code.
