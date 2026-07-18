# Readout Prior Decomposition — pre-registration (PM-directed; manuscript FROZEN; no amendment)

**Branch** `agent/cmi-trace-readout-prior-decomposition` (worktree `/home/infres/yinwang/CMI_AAAI_readout_prior`, base
`466a171f`). Only the owner stops/redirects a line. Central question: **is the anchoring win a genuine source-head
PRIOR value, or a weak-fresh-baseline / optimization-path / budget-mismatch artifact?** Only if source-centered MAP
beats a FAIR (hardened, converged, budget-matched) zero-centered ridge does "anchoring" become a real readout method.

## Narrowed statements (replace the two over-strong ones from the R-D report)
- NOT "the DG bottleneck is universally the readout" → `READOUT_HEADROOM_IS_DATASET_AND_BUDGET_DEPENDENT` (Lee2019
  near-ceiling, 2b few-shot harm).
- NOT "subspace confirmed non-causal" (all subspaces) → `TESTED_BCOND_ERASURE_NOT_CAUSAL_FOR_READOUT_GAIN` (current
  family only). Enough to keep the erasure/selection line PARKED; no new erasure methods.

## Fixed-prior-precision objective (P0.5)
All ridge/MAP heads use a FIXED prior precision τ on the SUMMED (not mean) CE, so shrinkage scales with 1/n:
`argmin_θ  Σ_i CE_i(θ) + τ ||θ − θ_anchor||²` (θ_anchor = 0 for ridge, θ_s for MAP). Equivalently
`mean CE + (τ/n)||θ−θ_anchor||²`. Small k → strong shrinkage (near frozen); large k → data-dominated (near refit).

## Readout arms (on the frozen source-standardised Z)
- **H0 frozen source head** θ=θ_s (no target labels).
- **H1 hardened zero-centered ridge** `argmin Σ CE + τ0||θ||²`; τ0 source-only, budget-matched. THE fair from-scratch baseline.
- **H1-W warm-start-matched zero-centered ridge** — SAME objective & τ0 as H1, init from θ_s. Convergence audit: must
  equal H1 to tolerance (convex objective); if not → `SOLVER_PATH_DEPENDENCE` (fix optimiser BEFORE any anchoring claim).
- **H2 source-centered Bayesian MAP** `argmin Σ CE + τs||θ−θ_s||²`; τs source-only, budget-matched.
- **H3 bias+temperature** (existing low-capacity control).
- **H4 source-only gated H2**: per outer fold & budget k, on source pseudo-target subjects compute δ_src(k)=U_H2−U_H0
  (pseudo-query); freeze gate g_k = 1[ mean δ − SE(δ) > 0 ]; deploy H4_k = H2_k if g_k else H0. Never sees target query.

## τ selection — budget-matched (P0.4), source-only
For each k∈{1,2,4,8,16,32,Full}, per source pseudo-target subject: draw k class-balanced from its EARLY session,
pseudo-query = its LATER session, SAME number of draws as the target experiment; select τ0(k) and τs(k) SEPARATELY per
k over grid {1e-3,1e-2,1e-1,1,10,100}. Save the FULL τ-curve (not just the winner) to separate zero-centered
regularization value / source-centered prior value / budget mismatch.

## Solver convergence audit (P0.3)
fit_ridge_map records res.success + final gradient norm. Init-invariance (H1 vs H1-W) is audited at the PARAMETER
level: ‖W_H1 − W_H1W‖ + ‖b_H1 − b_H1W‖ must be ≈0 (a strictly-convex ridge has a unique minimum). NOTE: the argmax
utility ΔU_init = U_H1-W − U_H1 is NOT a reliable init-invariance metric at large τ — heavy shrinkage makes logits
near-uniform, so argmax is knife-edge and flips on ~1e-5 numerical differences even when the parameters are identical
(verified: ‖W‖ diff 8.7e-6 yet 25% argmax disagreement at τ=10). Parameter-diff is authoritative; ΔU_init is reported
but a non-zero ΔU_init with matched parameters is argmax knife-edge, NOT SOLVER_PATH_DEPENDENCE.

## Primary endpoints (inference unit = target subject; draw→seed→subject; subject-cluster bootstrap + exact sign-flip; Holm across budgets)
1. **Prior-center value** `ΔU_center(k) = U_H2(k) − U_H1(k)` — does the source head as a prior CENTER beat ordinary
   ridge? (THE decisive question; primary.)
2. **Deployable utility** `ΔU_MAP-frozen(k) = U_H2(k) − U_H0`.
3. **Safe-policy utility** `ΔU_gate-frozen(k) = U_H4(k) − U_H0`; also `U_H4 − U_H2` (gate only avoids harm, or keeps gain?).
4. **Warm-start artifact** `ΔU_init = U_H1-W − U_H1` (must ≈ 0; else fix solver).
5. **Bias sufficiency** `U_H2 − U_H3`.

## High-powered matched-random subspace control (P6)
Sample random projectors CONTINUOUSLY until ≥50 matched controls (cap 5000; reason-code if unmet). Matching metric =
source-LOSO (or source-validation) bAcc — NOT in-sample. Match rank + source retention + source-head refit. Report
ΔG_h^specific for H1, H2, H4. If LCB95(ΔG_h^specific) ≤ 0 across datasets under high power → B_cond readout-causal
hypothesis stays PARKED.

## External headroom mechanism diagnosis (per target subject)
frozen source-head query bAcc + NLL; Full-cal target-head query gain; angle∠(W_s, W_t); bias vs direction
displacement; calibration Fisher/Hessian condition number; cal↔query mean/cov drift; class-prior shift; cal→query
frozen-loss gap; few-shot head parameter variance. Answer: Lee2019 = frozen≈oracle (no headroom)? 2b = full-cal
headroom positive but few-shot Fisher insufficient (direction-estimate variance > true shift)? — which Bayesian
sample-size scaling + the gate should address.

## Data matrix
All 4 (BNCI2014_001, BNCI2015_001, Lee2019_MI, BNCI2014_004): all subjects, seeds 0/1/2, 7 budgets, 50 draws, all
arms, native/informed/≥50-powered-matched-random. After the method+hyperparams FREEZE, add ≥1 LOCKBOX dataset NOT used
in any method decision, with a natural calibration→future-session axis (Cho2017 / High-Gamma candidates: generate a
session manifest; only cells with a natural session/run order enter; NO result-based dataset selection).

## Result routing (owner decides next)
- **P-A** `LCB95(U_H2 − U_H1) > 0` on ≥1 lockbox (others no harm) AND `U_H4 − U_H0 > 0` →
  `SOURCE_HEAD_PRIOR_IMPROVES_LABEL_EFFICIENCY` (real anchoring method).
- **P-B** H1 ≈ H2 but both > frozen → `GENERIC_REGULARIZED_TARGET_READOUT` (gain is regularization, not anchoring).
- **P-C** H1-W ≠ H1 → `OPTIMIZATION_PATH_ARTIFACT` (fix solver first; no anchoring claim).
- **P-D** gate avoids 2b harm but no positive utility → `SAFE_ABSTENTION_POLICY` (deploy-safe, not a performance method).
- **P-E** only Full positive → `READOUT_ADAPTATION_REQUIRES_DENSE_CALIBRATION`.
- **P-F** powered random control still no specificity → B_cond erasure stays PARKED.

## HOLD / FROZEN
New erasure / target-X selector / mechanism-consistency loss / learned projector / TTE = HOLD. Manuscript FROZEN.
Deliverables: this contract, `configs/cmi_trace_readout_prior_decomposition.yaml`, `tos_cmi/eval/readout_prior.py`,
`scripts/run_readout_prior_decomposition.py`, `scripts/aggregate_readout_prior_decomposition.py`, session manifests
(incl lockbox), tests, full real-EEG matrix. No amendment.
