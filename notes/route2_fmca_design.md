# Route 2 (FMCA + chain-rule) — Design Doc and Head-to-Head vs Route 1 (LPC-CMI)

## 0. TL;DR verdict

Route 2 belongs in the paper as an **appendix dependence-PROBE plus a documented Y-erasure ablation**, NOT as a competing training loss for the main result. The naive chain-rule (`min I(Z;S)`, `S=(D,Y)`) has a *structural* Y-erasure that better estimation cannot fix; the corrected difference `FMCA(Z,S) − μ·FMCA(Z,Y)` does fix it *in expectation* (verified on synthetic: ~0.3 at no-leak vs ~10 at leak) but is a high-variance, sign-unstable subtraction of two finite-sample-biased non-bound surrogates that re-introduces the min-max coupling FMCA was chosen to avoid. Route 1's posterior-KL conditions on Y *inside* the estimator, is a certified one-sided upper bound on `I(Z;D|Y)`, needs no matrix inverse/eigendecomposition, and empirically removes leakage at flat accuracy (`leakage_kl 0.528→0.011` at flat acc/label-sep on `BNCI2014_004/EEGNet`). Build Route 2 cheaply (~25–40 LOC estimator, encoder already done), report `FMCA(Z,D)` and `FMCA(Z,(D,Y))` as a leakage spectrum diagnostic, and present `fmca_chain`/`fmca_diff` as ablations that confirm *why* the conditioning-on-Y design (Route 1) is the right one.

---

## 1. How FMCA / NCD (Route 2) works

FMCA (Hu & Principe, "The Normalized Cross Density Functional," arXiv:2212.04631, v3 Feb 2024) estimates statistical dependence between two random variables U, V via the spectrum of the **Normalized Cross Density (NCD)** operator with kernel `K(x,x') = p(x,x') / [p^{1/2}(x) p^{1/2}(x')]`.

Mechanically, train two K-output networks `f_θ: U → R^K` and `g_ω: V → R^K`. For a minibatch of N paired samples with `F ∈ R^{N×K} = f(U)`, `G ∈ R^{N×K} = g(V)`, after centering:

- `R_F = F^T F / N` (K×K auto-correlation of f)
- `R_G = G^T G / N` (K×K auto-correlation of g)
- `P_FG = F^T G / N` (K×K cross-correlation)
- block `R_FG = [[R_F, P_FG],[P_FG^T, R_G]]` (2K×2K)

**Functional maximal correlations** `ρ_k` are the singular values of the whitened cross-correlation `P* = R_F^{-1/2} P_FG R_G^{-1/2}`; equivalently `λ_k = ρ_k²` are eigenvalues of `M = R_F^{-1/2} P_FG R_G^{-1} P_FG^T R_F^{-1/2}`. The **Total Statistical Dependence** read-out is

> **T_K = −(1/2) Σ_k log(1 − λ_k)**, with the top eigenvalue `λ_1 = 1` (a trivial constant mode) **discarded**, and a clamp `λ_k ← min(λ_k, 1−ε)`.

Two corrections to our internal note's stated form `−(1/K)Σ log(1−ρ_k²)`: the coefficient is **−1/2 (not −1/K)**, and the **top eigenvalue (=1) must be dropped**.

**Trainable surrogate (Eq. 14, no eigendecomposition required):** `r(f,g) = logdet R_FG − logdet R_F − logdet R_G = Σ_k log(1−λ_k) = −2·T_K`. Minimizing `r` *maximizes* dependence; to *minimize* dependence (our use) we add `+r` (equivalently minimize `−T_K`) to the loss. `torch.linalg.slogdet` backprops directly through this, so we never differentiate `eigh`/`svd` — this sidesteps the eigendecomposition-gradient instability. (The official notebook goes further with a smoothed-covariance trace surrogate `<A^{-1}, dA>`; for our small-K, batch-stratified setting the direct `slogdet` form is sufficient and simpler, and is exactly what the only EEG+FMCA repo, DY9910/MFMC, uses in `fmca_logdet_loss`.)

**HONESTY POINT — FMCA is NOT a calibrated MI estimator.** The authors explicitly position T_K as a *multivariate statistical dependence* functional, not Shannon MI ("MSD is multivariate, while MI is a scalar"). The identity `I = −(1/2)Σ log(1−ρ_k²)` holds *only in the jointly-Gaussian / linear-CCA case*. For general distributions T_K is a monotone dependence surrogate, **neither a proven upper nor lower bound on I**, and **not additive in nats**. This is the crux of the chain-rule problem (§3).

## 2. The exact FMCA(Z, S) estimator for continuous Z + discrete S

For discrete `S=(D,Y)` with `n = n_dom·n_cls` categories, this is the paper's FMCA-C case (§4.1): the discrete side is fed as the **one-hot indicator** `Y_oh ∈ R^{B×n}` (column-centered), optionally through a tiny affine map (the cost is affine-invariant). Then:

- `R_F = Φ^T Φ / B` (K×K, Φ = encoder/`f_θ`(Z) output)
- `R_G = Y_oh^T Y_oh / B` (n×n; **diagonal** with entries = class frequencies `p̂(s)` after centering — rank `n−1`)
- `P_FG = Φ^T Y_oh / B` (K×n)
- `M = R_F^{-1/2} P_FG R_G^{-1} P_FG^T R_F^{-1/2}` (K×K), eigenvalues `{ρ_k²}`
- `Î(Z;S) = −(1/2) Σ_{k=2}^{min(K, n)} log(1 − ρ_k²)` (drop the trivial top eigenvalue).

**Key structural reduction:** because `R_G` is diagonal in the class frequencies, `P_FG R_G^{-1} P_FG^T = Σ_s (1/p̂(s)) μ_s μ_s^T` where `μ_s` is the per-class mean embedding. So **FMCA(Z, discrete S) reduces exactly to a between-class-scatter (LDA-like) / canonical-correlation operator over the n category centroids** — its eigenvalues are the functional canonical correlations between Z and the class indicators. This makes the Y-erasure transparent (§3): Y *generates* those same one-hot directions, so the dominant eigenvectors of M are precisely the class-separating directions.

**Rank cap:** M has at most `n−1` informative eigenvalues; set `K ≥ n`. For BCI-IV-2a (n_cls=4) × ~8 train domains, `n≈32`, so ~31 informative correlations. Cheap, but it requires `B ≫ n` and a **(class×domain)-balanced sampler** so every S-cell is populated each step or `R_G` is singular and the estimate is ε-dominated noise. `ε·I` jitter (ε≈1e-3 to 1e-5) is mandatory on `R_F` and `R_G`; float64 for the covariances.

## 3. Chain-rule + Y-erasure analysis, and the chosen mitigation

**The bug in the ICML draft.** Eq. (8) uses `I(Z;D|Y) = I(Z;D,Y) − I(Z;Y)`, builds `S=(D,Y)`, then §4.4 drops the `+I(Z;Y)` correction term ("since CE maximizes I(Z;Y), just minimize I(Z;S)"), giving `L = CE + λ·FMCA(Z,S) + β·FMCA(Z,X_proj)`. This is a **structural identity, not an estimation artifact**: `λ·I(Z;S) = λ·I(Z;Y) + λ·I(Z;D|Y)` places weight λ *directly on the label channel*. CE (a saturating lower-bound surrogate on `I(Z;Y)`) and the FMCA penalty fight over the *same* `I(Z;Y)`, and for any λ large enough to suppress `I(Z;D|Y)` you also subtract comparable `I(Z;Y)`. This is exactly our observed collapse: the synthetic `chain` method (KL(q(S|Z)‖p(S)) ≈ I(Z;S)) drove label separability to ~35%. **Swapping MINE→FMCA changes the variance of the estimate, not the target of the optimization, so it cannot fix Y-erasure.**

**Candidate mitigations evaluated:**

- **(a) Corrected difference `FMCA(Z,S) − μ·FMCA(Z,Y)`.** This is the right object in expectation (the chain-rule identity at μ=1) and **does** restore the protective term. We verified on a controlled synthetic (N=600, nY=2, nD=3): no-leak → `6.08 − 5.79 = 0.30 (≈0, correct)`; leak → `16.01 − 5.82 = 10.19 (large, correct)`. So at the safe point the two terms cancel and label info is not subtracted. **But** it is a small difference of two large, *separately-biased*, *non-bound* surrogates with different K-truncation, ε-bias, and rank caps (n vs n_cls) → catastrophic-cancellation variance; the difference can go negative (rewarding *increasing* leakage); the `−μ·FMCA(Z,Y)` gradient can degenerately push Z to *increase* `I(Z;Y)`; and you re-introduce a min-max coupling (maximize one FMCA head, minimize the other through a shared encoder). No FMCA/NCD paper validates a conditional/difference form, and a downstream FMCA paper (arXiv:2512.23076) reports the log-det objective diverging after ~13k steps (−12.4 pts vs a trace surrogate).

- **(b) Project Y out first (CIRCE-style residualization).** Theoretically clean and batch-size-robust (Y-info regressed away before the penalty, provably zero iff `Z ⟂ D | Y`). Already wired as `circe` in the harness. But this is no longer FMCA; it is a different estimator that converges on Route 1's design philosophy.

- **(c) Stratified per-class FMCA — `mean_y FMCA(Z, D | Y=y)`** using the `n_dom`-way one-hot D within each class. Cannot erase Y by construction (Y held fixed per stratum). This is the most defensible *FMCA-native* conditional form. Cost: small per-stratum batches → severe per-class estimation noise (the exact problem §4.4 of the draft tried to dodge).

**Chosen mitigation for the harness:** implement BOTH `fmca_chain` (the naive S-penalty, to *reproduce* and *document* Y-erasure as an ablation) and `fmca_diff` (the corrected difference `FMCA(Z,S) − μ·FMCA(Z,Y)`, with `μ`-balancing and a `clamp-at-0` on the difference, as the cautionary higher-variance baseline). Use the encoder-agnostic flat Z (EEGNet *or* TSMNet) since FMCA is geometry-blind. Do NOT promote either to the main objective.

## 4. Which encoder

**FMCA-chain does NOT require SPDNet/TSMNet.** FMCA is encoder-agnostic: it consumes arbitrary real feature vectors via the `f_θ` net; its log-det self-whitens the *network outputs* (K-dim), not the input Z, and imposes no SPD/whitening/dimensionality constraint on Z. In the draft, every FMCA argument is *already* a flat Euclidean vector (LogEig tangent Z, one-hot S, tangent X_proj) — FMCA runs strictly downstream of the Riemannian part. The SPDNet coupling is a separable EEG-geometry choice ("swelling effect"), not a mathematical dependency; no prior work combines FMCA with SPD features, so it would be only trivially novel.

**Practical choice:** use the already-integrated **TSMNet backbone** (`repos/TSMNet`, BSD-3-Clause, GPU-patched) as the optional representational upgrade — it returns a flat `tsdim = subspacedims·(subspacedims+1)/2 = 210`-dim LogEig tangent Z (`TSMNetBackbone.z_dim`), which is the continuous Z for FMCA. The *one* narrow place Route 2's estimator has a genuine edge is here: tangent-space LogEig features are plausibly closer to Gaussian, where T_K is closest to true MI. But the encoder is orthogonal to the Y-erasure question and cannot rescue it. EEGNet works identically as the FMCA encoder for the ablation rows.

## 5. How it plugs into our harness (new methods)

The harness already has the exact seams. The two-step alternating trainer (`cmi/train/trainer.py`: Step A fits `post` on detached Z via `post.posterior_loss`; Step B does `loss = CE + lambda_t * post.reg(cmi_method, z, yb)`) and `DomainPosteriors.reg(method, z, y)` (`cmi/methods/regularizers.py`) are the plug-in point. FMCA's `f_θ` net (and an optional `g_ω` for the S side, though one-hot can be used raw) live as extra submodules of `DomainPosteriors`; their parameters are trained in Step A alongside the posteriors (on detached Z), and the FMCA term is evaluated on grad-carrying Z in Step B inside `reg`.

New `METHODS` entries:
- **`fmca_chain`**: `reg` returns `+T_K(f_θ(z), onehot(S))` (minimize dependence on the joint S). Reproduces Y-erasure; the documented ablation.
- **`fmca_diff`**: `reg` returns `T_K(f_θ(z), onehot(S)) − μ·T_K(f_θ(z), onehot(Y))` (corrected chain rule), with `μ∈[0.5,1]` and the difference clamped ≥0. The cautionary corrected baseline.
- **(optional) `fmca_strat`**: `mean_y T_K(f_θ(z), onehot(D)|Y=y)` — the FMCA-native conditional form, if a (class×domain)-balanced sampler is enabled.

For the **probe** (recommended primary use): a non-training diagnostic that, at eval time, reports `FMCA(Z,D)` and `FMCA(Z,(D,Y))` spectra `{ρ_k}` as a richer leakage figure alongside the scalar `leakage_kl` — slots into `cmi/eval/leakage_audit.py`.

## 6. Head-to-head plan vs Route 1 (LPC-CMI)

Run all on the same backbone, sampler, λ-grid, and seeds so only the *objective* differs.

1. **Leakage-removal vs label-preservation curve (the decisive plot).** λ-sweep for `lpc_prior` (Route 1), `fmca_chain`, `fmca_diff`, `erm`, on `BNCI2014_004/EEGNet` and `BNCI2014_001/EEGNet`. Axes: `leakage_kl` (and the FMCA `{ρ_k}` spectrum) vs target accuracy and label-separability. Expectation, grounded in the existing `BNCI2014_004_EEGNet_lamsweep.json`: `lpc_prior` drives leakage down ~48× at flat accuracy/label-sep; `fmca_chain` collapses label-sep (Y-erasure) as λ grows; `fmca_diff` holds label-sep better than `fmca_chain` but with markedly higher run-to-run variance and occasional negative-difference instability.
2. **Variance / stability table.** Per-seed std of target acc and of the CMI/dependence estimate; count of divergence/negative-difference events; wall-clock. Route 1 (smooth softmax, no matrix inverse) vs Route 2 (slogdet, ε-jitter, eigenvalue clamp).
3. **Gaussianity-edge probe (the one place Route 2 might win).** Repeat (1) on the **TSMNet tangent Z** and report whether `fmca_diff`'s near-Gaussian regime narrows the gap to `lpc_prior`. Report the `{ρ_k}` spectrum as a leakage diagnostic regardless.
4. **Estimator-quality cross-check.** On synthetic with known `I(Z;D|Y)`, compare Route 1's posterior-KL upper bound vs `fmca_diff` vs `fmca_strat` for bias/variance against ground truth.

Disposition: Route 1 = main method (tables, all datasets); Route 2 = appendix (probe figure + Y-erasure ablation + the one Gaussianity-edge experiment).

## References (findings + repos)
- NCD/FMCA canonical: **arXiv:2212.04631** (Hu & Principe). No official Hu/Principe code exists (verified). Estimator must be reimplemented (~25–40 LOC).
- Only EEG+FMCA code: **github.com/DY9910/MFMC** (arXiv:2512.23076) — *unlicensed / all-rights-reserved*; READ-ONLY math/structure reference, do NOT vendor. Useful facts: K=128, B=200, float64, eps=1e-5, cov_beta=0.5; documents log-det divergence vs trace surrogate.
- Official NCD notebooks: **github.com/bohu615/density-ratio-decomposition** — *no LICENSE = all-rights-reserved*; reference only.
- Encoder: **github.com/rkobler/TSMNet** (BSD-3-Clause, safe to vendor; already integrated, returns 210-d LogEig tangent Z).
- Route-1 / conditioning baselines: CLUB (arXiv:2006.12013, MIT, `github.com/Linear95/CLUB`); CIRCE (arXiv:2212.08645, MIT, `github.com/namratadeka/circe`, already wired as `circe`); IIB (arXiv:2106.06333); McAllester–Stratos (arXiv:1811.04251) — confirms upper-bound minimization is the sound side for a *suppression* objective.