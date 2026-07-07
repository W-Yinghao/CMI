# Route 2 — FMCA + chain-rule CMI (from the old ICML draft)

Source: `icml2026 (1).pdf` ("Conditional Mutual Information for Causality-Invariant EEG Domain Generalization").
This is the route the AAAI plan **deprecated** in favor of route-1 (LPC-CMI posterior-KL). User wants it explored as a
**second, parallel route**. Design workflow: `wzdwx71mr` (running) → output lands in this note.

## The method
1. **Riemannian SPDNet encoder**: EEG → spatial covariance → BiMap (`P_l = W_l P_{l-1} W_l^T`) → ReEig (eigenvalue clamp)
   → LogEig (tangent) → vectorized `Z`. **We already have this** as the `TSMNet` backbone (GPU-runnable).
2. **Objective (IIB-style)**: `max_θ  I(Z;Y) − λ·I(Z;D|Y) − β·I(Z;X)`. The invariance term `I(Z;D|Y)` is the **same conditional
   as our LPC-CMI** (note the draft cites IIB but uses I(Z;D|Y), not IIB's I(Y;D|Z)).
3. **Chain-rule**: `I(Z;D|Y) = I(Z;D,Y) − I(Z;Y)`; super-variable `S=(D,Y)`; "since CE maximizes I(Z;Y), just minimize I(Z;S)".
4. **FMCA estimator** (Hu & Principe 2022, NCD, arXiv 2212.04631; paper saved `papers/NCD_FMCA_HuPrincipe_2212.04631.pdf`):
   `Î(U;V) ≈ −(1/K) Σ_k log(1 − ρ_k²)`, ρ_k = functional maximal correlations = eigenvalues of the Normalized Cross Density
   operator. Log-det, least-squares-like; claimed more stable than MINE (no exp) and HSIC (no fixed kernel bias).
5. **Loss**: `L = CE(g(Z),Y) + λ_inv·FMCA(Z,S) + β·FMCA(Z,X_proj)`.

## My critical analysis (fed to the workflow)
- **Same target as route-1's failure mode.** Minimizing `I(Z;S)=I(Z;D,Y)` reduces `I(Z;Y)+I(Z;D|Y)` → it **fights CE and erases Y**.
  We **already demonstrated this**: the synthetic `chain` method (minimize KL(q(S|Z)‖p(S)) ≈ I(Z;S)) collapsed label separability to ~35%.
  FMCA estimates the *same* I(Z;S), so a better estimator does **not** by itself fix Y-erasure.
- **Open question for route-2:** does a **corrected chain-rule** (`FMCA(Z,S) − μ·FMCA(Z,Y)`, an explicit difference) or
  **stratified/per-class FMCA** avoid Y-erasure — or is route-1 (conditioning directly on Y) fundamentally safer? The difference-of-
  noisy-estimators may be high-variance.
- **Encoder is ready** (TSMNet/SPDNet). So route-2 ≈ TSMNet backbone + a new `FMCA(Z,S)` loss module.

## Workflow deliverables (pending)
FMCA/NCD exact algorithm + official code; FMCA for continuous-Z vs discrete-S; chain-rule Y-erasure mitigation verdict;
encoder fit; route-1-vs-route-2 theory; integration plan + `fmca_module_spec`; downloads. Decision: route-2 as
main-method / complementary / **appendix dependence-probe** (the AAAI plan's original disposition).
