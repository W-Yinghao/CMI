# MCC estimator audit — two-pass EXACT population-gradient discriminator (SPEC; NOT an amendment; manuscript FROZEN)

Cheap, no-training diagnostic authorized by PM 2026-07-18 after MCC-λ1 = SPECIFIC_BUT_LAMBDA_INERT (DG-null).
Branch `agent/cmi-trace-mcc-estimator-audit`, base a8c298ea. Separates the two explanations of the tiny λ-inert
global-MCC geometry effect: **K=4 episodic estimator variance/bias** vs **the global equal-weight consistency
TARGET is wrong**. Only the project owner may stop a scientific line.

## Scope discipline (PM)
A FROZEN (no-training) diagnostic can ONLY test whether the K=4 episodic MCC gradient is a poor estimate of the
exact full-source MCC gradient. It CANNOT prove "geometry → DG"; DG payoff waits for a full training experiment
AFTER the audit passes. Do NOT require the diagnostic to show geometry tracks DG. Default next fork = risk-weighted
MCC; EMA/prototype only if the audit directly proves K=4 is variance-limited.

## Why two-pass, not a giant batch
EEGNet has BatchNorm. A giant raw-input batch would confound prototype-estimation variance with BatchNorm
batch-statistics change. Instead compute the EXACT full-source MCC gradient w.r.t. the CURRENT encoder params
without a giant batch or storing the whole source graph, with BN running-stats FROZEN (model.eval()):

**Pass 1 — exact source prototypes + prototype gradient.** Forward all continuation-train source in micro-batches
(eval, BN frozen), collect z_i (detached). Per (subject d, class c): μ_{d,c} = (1/n_{d,c}) Σ_{i∈cell} z_i. Make the
μ's leaf tensors (requires_grad), compute the population MCC loss L_MCC^all on them, backprop → g^μ_{d,c} =
∂L_MCC^all/∂μ_{d,c}.

**Pass 2 — exact backprop to the encoder.** Since μ_{d,c} = (1/n_{d,c}) Σ_i z_i, each sample's upstream feature
gradient is ∂L/∂z_i = g^μ_{cell(i)} / n_{cell(i)}. Re-forward each micro-batch WITH grad (eval, BN frozen), call
z_batch.backward(gradient = upstream_batch), ACCUMULATING into θ.grad across micro-batches. Result:
g_θ^full = ∇_θ L_MCC^all — exact, because in eval mode BN uses frozen running stats so z_i is an independent
function of x_i (no batch coupling) and per-sample micro-batching is exact. (In TRAIN mode BN couples samples and
this would NOT be exact — eval mode is REQUIRED and asserted.)

## The 4 estimators (per cell)
1. **Exact population** g_θ^full (two-pass, all continuation-train source).
2. **Current estimator** K=4 balanced episodes, R≥64 independent draws → g_θ,4^(1..R).
3. **Large-batch** K=16 (or the max feasible for a short cell, reason-coded) → g_θ,16^(1..R).
4. **Shuffled-subject control** at the K=4, K=16, and population-prototype levels (same MCC formula on
   within-class-permuted subject labels). The shuffle is NOT a no-op (it mixes subjects → smoothing); the audit
   must confirm the full-source TRUE vs full-source SHUFFLED objectives are functionally different gradients.

## Primary diagnostics (θ-level unless noted)
- Gradient alignment  A_K = cos( mean_r g_θ,K^(r), g_θ^full ).
- Relative bias       B_K = ‖ mean_r g_θ,K^(r) − g_θ^full ‖₂ / (‖g_θ^full‖₂ + ε).
- Gradient SNR        SNR_K = ‖mean_r g_θ,K^(r)‖₂² / ( (1/R) Σ_r ‖g_θ,K^(r) − mean‖₂² + ε ).
- Normalized one-step PROTOTYPE effect: normalize each estimator's PROTOTYPE gradient to unit norm, take a step
  M' = M − α·(g^μ/‖g^μ‖) in prototype space, measure ΔWSCI_full, ΔWSCI_{K=4}, ΔWSCI_{K=16} (WSCI = source
  direction-consistency on prototypes). Answers: is the K-vs-full gap from gradient SCALE or DIRECTION?
- True-vs-shuffle separation: ‖g_true − g_shuffle‖, cos(g_true, g_shuffle) at the full θ-level.

## Matrix
63 fold-seed cells = (9+12) subjects × 3 seeds × 2 datasets, full LOSO. PRIMARY checkpoint = each bundle's ERM
warm-up (where continuation starts — the point where a misleading episodic estimator would first bite). SECONDARY
(no retraining): reuse the existing λ0.25 and λ1.0 A/B/C feature dumps for a feature/prototype-level end-of-training
audit. NO model training; NO 189-arm fleet.

## Routing (simple; frozen YAML, no amendment)
- **E1 — estimator variance-limited** (ALL): both-dataset MEDIAN A_4 < 0.5; AND A_16 − A_4 > 0.25; AND the exact
  full-gradient normalized population-WSCI movement ≥ 2× the K=4 mean-gradient's; AND not driven by 1–2 subjects →
  label `EPISODIC_MCC_ESTIMATOR_VARIANCE_LIMITED` → approve ONE source-only EMA / memory-bank prototype MCC round
  (then a full A/B/C GPU matrix to test DG).
- **E2 — K=4 already estimates the full gradient well** (e.g. A_4 ≥ 0.8, K=16/full no significant normalized
  population-movement gain) → global MCC is not minibatch-misled; it optimizes a DG-irrelevant geometry →
  **risk-weighted MCC** (NOT EMA).
- **E3 — the full gradient itself is weak / near shuffle** → even a zero-variance estimator lacks independent
  training signal for unweighted global consistency → **risk-weighted MCC**.
- **Mixed** (datasets disagree) → add EPISODE DRAWS to cut Monte-Carlo uncertainty (NOT new seeds, NOT training);
  still inconsistent → default to risk-weighted (geometry–DG decoupling is the stronger standing evidence).

## Deliverables (this branch; NOT a prereg amendment)
`tos_cmi/eval/mcc_estimator_audit.py`, `scripts/run_mcc_estimator_audit.py`, `scripts/aggregate_mcc_estimator_audit.py`,
`scripts/sbatch_mcc_estimator_audit.sh`, `configs/cmi_trace_mcc_estimator_audit.yaml`, this plan.

## Tests (8, pinned)
1 two-pass g_θ == a small-fixture single-graph full-batch g_θ (numerical, eval mode); 2 K=all episodic prototype
gradient == the population prototype gradient; 3 BN running stats byte-identical before/after the audit; 4 target
arrays never enter any loss/gradient (signature + run check); 5 cell-specific random seeds; 6 true AND shuffle both
saved; 7 63-cell completeness fail-closed; 8 warm-up hash consistent with the existing MCC rounds.

## Execution order
1) validate two-pass gradient on ONE real EEG cell; 2) then auto-submit the 63-cell full audit; 3) aggregate +
report; 4) do NOT start EMA / risk-weighted training or a new 189-arm GPU fleet. HELD: EMA, risk-weighted, M2,
projector, TTE, CMI, manuscript.
