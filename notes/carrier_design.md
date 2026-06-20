# Tri-CMI Carrier & Estimator Design (AAAI-27)

**Verified against the live environment (conda env `icml`, 2026-06-06):** torch 2.8.0+cu128, braindecode **0.8**, moabb 1.2.0, pyriemann 0.7, geoopt 0.5.1, skorch 1.2.0, sklearn 1.5.2. All carrier models below were import- and forward-pass-tested in this env. The synthetic sanity check (`/home/infres/yinwang/CMI_AAAI/synthetic/sanity_check.py`) already implements and validates the exact LPC-CMI estimator plus its four ablations — the MOABB work is the remaining build.

---

## 0. Material corrections to the findings (verified)

These affect the build plan and must be fixed before committing:

1. **MSVTNet and Labram (LaBraM) are NOT in braindecode 0.8.** `from braindecode.models import MSVTNet` and `... import Labram` both raise `ImportError` in the pinned env. The findings' "score 4/5, present in braindecode" claims for MSVTNet and the `Model.from_pretrained(...)` foundation-model convenience API are **braindecode >= 1.4 features**. Under the pinned 0.8 stack these are unavailable. **Decision:** drop MSVTNet and all four EEG foundation models (LaBraM/CBraMod/EEGPT/BIOT) from the must-have tier. If we want them, that is a *separate, optional* env (`eeg2025` has moabb 1.5 but **no braindecode** — would need a fresh braindecode 1.5 install), explicitly out of the AAAI core.
2. **EEGConformer `return_features=True` is the cleanest Z extractor available.** Verified: `EEGConformer(n_outputs=4, n_chans=22, n_times=1000, return_features=True)(x)` returns `(logits[B,4], features[B,32])`. The 32-dim token *is* the representation `Z` — no forward hook, no surgery. This makes Conformer the easiest architecture-agnosticism demonstrator, not EEGNet.
3. **The core estimator is already built and validated.** `sanity_check.py` method `lpc_prior` is exactly `E_i KL(q_psi(D|z_i,y_i) || pi_{y_i}(D))` with the CLUB-style two-step alternation, Laplace-smoothed `pi_y` (`empirical_priors(..., alpha=1.0)`), warmup, and a frozen-encoder leakage probe. The four ablations (`erm`, `marginal`, `chain`, `lpc_uniform`) are the precise failure-mode contrasts. **Protocol D is essentially done** — only needs a results table written up. Reuse `kl_to_prior`, `empirical_priors`, the alternating optimizer, and the `leakage_probe` verbatim for the EEG harness.

---

## 1. Recommended carrier stack (encoder × framework)

### 1.1 Primary encoder: EEGNetv4 (braindecode 0.8)
- **Why:** verified `EEGNetv4(n_chans=22,n_outputs=4,n_times=1000)(x)->[B,4]`, ~tiny params, BSD-3. The low-dim **F2 bottleneck** (default 16) is the ideal continuous `Z` for `q_psi(D|z,y)`: low dimension ⇒ the domain posterior is cheap and well-conditioned, and the KL-to-prior is stable. Tiny capacity also limits trivial subject/session memorization, which is the right inductive bias when minimizing `I(Z;D|Y)`.
- **Z extraction:** forward hook on the layer *before* the final conv-classifier/flatten (the F2-channel feature map, global-pooled to a 16-or-32-vector). Recommend **bumping F2 to 32** so `Z` carries enough `Y`-info that the regularizer cannot trivially collapse the task subspace.
- **Risk to manage:** small capacity can underfit hard cross-dataset shifts → keep F2=32 as default for cross-dataset (Protocol C).

### 1.2 Primary baseline encoder: ShallowFBCSPNet (braindecode 0.8)
- **Why:** the canonical MOABB/BNCI MI baseline (~46k params), official trialwise recipe documented (AdamW, `lr=0.0625*0.01`, `weight_decay=0`, `batch_size=64`). Physiologically meaningful band-power features generalize across subjects → fair "does LPC-CMI preserve Y" demonstrator.
- **Z extraction caveat:** pre-classifier feature is a *flattened conv map*, not a clean vector. Add a `nn.AdaptiveAvgPool` + small linear projection to a 16/32-dim `Z` head before `q_psi`.

### 1.3 Architecture-agnosticism proof: EEGConformer + ATCNet (braindecode 0.8)
- **EEGConformer:** `return_features=True` → `(logits, features[B,32])`, the easiest plug-in of all (verified). Shows the regularizer is not conv-specific. Upstream repo GPL-3.0, but **use the braindecode reimplementation (BSD-3)** to avoid copyleft.
- **ATCNet:** `ATCNet(n_chans, n_outputs, ...)`, Apache-2.0 upstream, braindecode-native, SOTA-ish on BCI-IV-2a. Tap the post-attention/TCN pooled embedding as `Z`.

### 1.4 High-capacity stress test: Deep4Net (braindecode 0.8)
- ~283k params (~80× EEGNet). The most likely to memorize subject identity ⇒ the most interesting test of whether `I(Z;D|Y)` suppresses leakage *as capacity grows* without destroying `Y`. Needs KL **warmup** and a separate lr/weight_decay (per the official tutorial). Use as an ablation that demonstrates the regularizer's effect scales with backbone capacity.

### 1.5 Optional geometric ablation: LogCov + Tangent Space (pyRiemann 0.7)
- `Covariances(estimator='oas') -> TangentSpace(metric='riemann')` gives a **deterministic, well-conditioned Euclidean `Z`** — the safest substrate for the `q_psi` posterior and the post-hoc leakage probe. **Frozen-feature only** (not trainable end-to-end); pair with per-domain recentering for DG. Doubles as the dominant calibration-light MI reference accuracy.
- **Avoid trainable SPD nets as core** (SPDNet/TSMNet/EEGSPDNet): eigen-decomposition backward passes are gradient-unstable — exactly what we want to avoid. TSMNet stays only as a **baseline** (§3).

### 1.6 DG-framework wrapper
- Use **braindecode's `EEGClassifier` (skorch)** for the standard task loop, but the LPC-CMI term needs the **two-step alternation** (posterior step on detached `z`, then encoder+head step), so the cleanest path is a **custom PyTorch training loop** mirroring `sanity_check.py.train_one` rather than fighting skorch callbacks. The unified harness (§4) is framework-agnostic: backbone → `Z` → `{task head, q_psi head}`.

---

## 2. Recommended CMI estimator + ablation variants

### 2.1 Core: LPC-CMI (variational posterior-KL upper-bound surrogate)
`L_CMI = (1/N) Σ_i KL( q_psi(D | z_i, y_i) || pi_{y_i}(D) )`

- **q_psi:** a **CLUBForCategorical**-style softmax classifier taking `[z, one-hot(y)]` → logits over `D` (continuous `Z`, discrete `D`). In `sanity_check.py` this is `q_dzy = Head(h+2, n_dom)`; for EEG it is a 2-layer MLP on the F2/token `Z` plus one-hot `Y`.
- **pi_y(D)=p(D|Y):** Laplace-smoothed (`alpha=1.0`), computed once per training set from `(Y,D)` counts (`empirical_priors`). Laplace smoothing is essential under LOSO where `(Y,D)` strata can be thin.
- **Training:** CLUB two-step alternation (already coded): **Step A** fit `q_psi` by cross-entropy on `enc(x).detach()`; **Step B** update encoder+task-head with `CE(clf(z), y) + lambda_t * L_CMI`, `lambda_t` warmed up over the first ~15 epochs.
- **Why this and not the alternatives:**
  - **Upper bound, correct direction.** By Barber-Agakov, `I(Z;D|Y) = H(D|Y) - H(D|Z,Y)`; `H(D|Y)` is the `pi_y` entropy and `H(D|Z,Y)` is the `q_psi` cross-entropy, so minimizing the KL genuinely pushes true leakage down. Lower bounds (MINE/InfoNCE/DV) are the **wrong direction** for a minimization penalty and are statistically capped at `O(ln N)` (McAllester-Stratos) and high-variance.
  - **Non-adversarial, plug-in, stable.** No GRL, no generator, no min-max equilibrium — the design stance shared with Moyer et al. (non-adversarial conditional invariance).
  - **Protects Y by construction.** Conditioning the prior/posterior on the *true* label `Y` (not erasing it) is what distinguishes us from marginal `I(Z;D)` (label-erasing) and super-label `I(Z;(D,Y))` (Y-erasing).
- **Licensing:** the Linear95/CLUB repo has **no license** → reimplement the ~25-line `CLUBForCategorical` in our own MIT code (already effectively done in `sanity_check.py`). Do not vendor.

### 2.2 In-paper ablations (all already in or trivially added to `sanity_check.py`)
| Variant | Objective | Role / expected failure |
|---|---|---|
| `erm` | task CE only | over-relies on spurious/domain shortcut |
| `marginal` | `E KL(q(D\|Z) \|\| p(D))` ≈ `I(Z;D)` | **label-erasure** under imbalance (DANN/CORAL/MMD class) |
| `chain` | `E KL(q(S\|Z) \|\| p(S))`, `S=(D,Y)` | **Y-erasure** → chance accuracy |
| `lpc_uniform` | `E KL(q(D\|Z,Y) \|\| Uniform)` | **the CDANN target**; mis-specified under non-uniform within-class domain proportions |
| `lpc_prior` | `E KL(q(D\|Z,Y) \|\| pi_y(D))` | **ours** — keeps causal feature, drops domain leakage |

The `lpc_uniform` vs `lpc_prior` contrast is the cleanest isolation of our central claim (empirical `pi_y` target beats the uniform target that CDANN implicitly uses), and the MOABB experiment should report both.

### 2.3 Estimator probes / diagnostics (not the trainable loss)
- **CIRCE** (official MIT PyTorch, ICLR'23 Oral): the principal **kernel-side conditional-independence** competitor (`Z ⊥ D | Y`), and a candidate *alternative carrier*. Precompute-once design avoids the per-step `O(m^3)` Gram inverse that makes HSCIC/Fukumizu fragile on small EEG batches.
- **Class-stratified HSIC** (a few lines): cheap kernel baseline — for discrete `Y`, conditional HSIC collapses to `Σ_y p(y) · HSIC(Z, D | Y=y)`.
- **Conditional (Y-stratified) InfoNCE:** the label-protecting *lower-bound* twin to bracket the LPC-CMI upper bound in the paper.

### 2.4 Post-hoc leakage certification (offline, non-differentiable)
- **Frozen-encoder probe** (already in `sanity_check.py.leakage_probe`): train `q_probe(D|Z,Y)` on a held-out source split, report mean `KL(q_probe || pi_y)` and `leakage_advantage = cond_dom_acc - prior_dom_acc` (domain accuracy above what `Y` alone gives). This is the headline "conditional domain leakage" metric.
- **Permutation HSIC test per Y-stratum:** calibrated p-value that residual `I(Z;D|Y) > 0`.
- **knncmi** (Mesner-Shalizi mixed kNN CMI, GPL-3.0): trusted **offline ground-truth** `I(Z;D|Y)` on Protocol-D synthetics (Z continuous, D,Y discrete). Keep as an isolated eval-only dependency (never vendor GPL into release).
- **sklearn `mutual_info_classif`** (Ross, BSD, in-stack): cheap marginal-MI surrogate; stratify per `Y` for an approximate mixed-data CMI.

---

## 3. Baselines (priority order)

**Tier-1 (headline comparisons):**
- **CDANN** (DomainBed, MIT) — *single closest comparison*. Conditions discriminator on `Y`, pushes domain posterior to **Uniform**. Running CDANN vs `lpc_prior` isolates two axes: (a) uniform target vs `pi_y(D)` target; (b) adversarial min-max vs adversary-free KL. This is the most important experiment.
- **EEG-DG** (XC-ZhongHIT, no license → reproduce) — same MI-EEG calibration-free DG task, same datasets (2a/2b/OpenBMI), reusable Modified-EEGNet backbone. Head-to-head accuracy/kappa to beat (reports 81.79%/0.7572 on 2a, 87.12%/0.7424 on 2b).
- **IIB** (Luodian/IIB, MIT) — theoretical foil: minimizes `I(Y;D|Z)+I(X;Z)` (reversed variable ordering vs our `I(Z;D|Y)`). Headline related-work distinction + swap-in ablation.

**Tier-2 (standard DG):**
- **DANN** (DomainBed, MIT) — marginal `I(Z;D)` lower-baseline that motivates conditioning.
- **Deep CORAL + MMD + C-CORAL/C-MMD** (DomainBed penalty fns, MIT) — moment-matching baselines; the conditional C-CORAL/C-MMD are the "cheap conditional" competitors our learned posterior must beat (plug-in moments are high-variance with few trials per `(D,Y)` cell).
- **VREx** (most stable risk-based) + **GroupDRO** (worst-subject robustness) — DomainBed classes. Skip/deprioritize IRM (most fragile, rarely beats ERM).

**Tier-3 (geometric / contrastive, optional):**
- **TSMNet/SPDDSMBN** (rkobler/TSMNet, BSD-3, geoopt+moabb compatible) — strongest geometric DG baseline; removes `D` marginally/implicitly (motivating contrast).
- **SCLDGN SupConLoss** (hongyizhi/SCLDGN, no license → reimplement) — label-protecting contrastive counterpart.

---

## 4. Unified-harness plug-in design (how LPC-CMI attaches to any encoder)

The regularizer is a **drop-in auxiliary loss on a continuous `Z` plus discrete `Y,D`**. Three pieces, none touching the backbone:

```
forward:  Z = f_theta(X)          # backbone penultimate feature (hook OR return_features)
          logits_y = task_head(Z) # standard classifier
          logits_d = q_psi([Z, onehot(Y)])   # domain posterior (2-layer MLP), trained on Z.detach()

loss (Step A, posterior):  CE(q_psi([Z.detach(), onehot(Y)]), D)
loss (Step B, encoder):    CE(task_head(Z), Y)
                         + lambda_t * mean_i KL( softmax(logits_d_i) || pi_y[Y_i] )
```

**`get_Z(model, X)` adapter table (verified):**
| Carrier | How to get Z | Z dim |
|---|---|---|
| EEGNetv4 | forward hook before final conv/flatten + global-pool | F2 (16→32) |
| ShallowFBCSPNet | hook on pre-classifier map + AdaptiveAvgPool + linear | proj (32) |
| EEGConformer | `return_features=True` → `(logits, features)` | 32 (verified) |
| ATCNet | hook on post-TCN pooled token | model-dependent |
| Deep4Net | hook before final conv classifier + pool | proj (32) |
| LogCov+TS (frozen) | precomputed tangent vectors | C(C+1)/2 |

**Reuse directly from `sanity_check.py`:** `kl_to_prior(logits, log_prior)`, `empirical_priors(y, d, n_dom, alpha)` (returns `pi_y, p_d, p_dy` for all four ablations), the two-optimizer alternation in `train_one`, and `leakage_probe`. The only EEG-specific additions are (a) the `get_Z` adapters, (b) MOABB LOSO/cross-session/cross-dataset data loaders (use `cmi/paths.py.configure_offline_moabb()` for offline cache), and (c) the metric suite (BalAcc, Macro-F1, Worst-Subject Acc, ECE/NLL, leakage KL/advantage, label separability).

**Domain `D` definition per protocol:** A (LOSO) `D`=subject; B (cross-session) `D`=subject×session; C (cross-dataset) `D`=dataset (and/or device). `q_psi` output dim = number of *source* domains; `pi_y` recomputed per split.

---

## 5. Risk table

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Upper-bound looseness: `q_psi` underfits → KL not a valid bound | High | Med | Inner posterior lr 2× encoder lr (as in `sanity_check.py`: 2e-3 vs 1e-3); enough Step-A updates per Step-B; report frozen-probe leakage as the real audit, not the in-loop KL. |
| KL term collapses task subspace (erases Y) | High | Low (conditioning protects Y) | `lambda` warmup; F2=32 so Z has Y-capacity; monitor `label_sep` (linear probe) every epoch; the `lpc_uniform`/`chain` ablations bound the failure. |
| Thin `(Y,D)` strata in LOSO → degenerate `pi_y` | Med | High | Laplace smoothing `alpha=1.0` (already in `empirical_priors`); for cross-dataset, collapse rare domains. |
| MSVTNet/LaBraM unavailable in braindecode 0.8 | Med | Certain (verified) | Dropped from core; foundation-model carriers are explicitly out-of-scope unless a separate braindecode>=1.4 env is built. |
| Deep4Net high-capacity instability with KL term | Med | Med | Warmup + gradient scaling on the CMI term; separate lr/weight_decay. |
| Baseline-repo licensing (CLUB, EEG-DG, SCLDGN, CDAN, ManyDG = no license) | Med | Certain | Reimplement short loss classes in our own MIT code; cite papers; contact authors before vendoring. knncmi GPL → eval-only, isolated. EEG-Deformer CBCR non-commercial → reproduce-only, do not redistribute. |
| Adversarial baselines (DANN/CDANN) unstable on small EEG batches | Low | Med | Use DomainBed alternating-optimization variant (no GRL); tune `lambda` schedule; report stability honestly vs our adversary-free term. |
| MMD/C-MMD high variance with few samples per `(D,Y)` cell | Low | Med | Multi-kernel + larger batches; frame variance as the argument for the learned posterior. |
| skorch/`EEGClassifier` cannot express two-step alternation cleanly | Low | Med | Use a custom loop mirroring `train_one`; skorch only for plain ERM baselines. |

---

## 6. Tiered build order (AAAI: abstract 2026-07-21, full paper 2026-07-28)

### MUST-HAVE CORE (the paper does not exist without these)
1. **Protocol D write-up** — `sanity_check.py` already runs; add knncmi ground-truth column on synthetics, finalize the 5-method table. *(~done; ~1 day)*
2. **MOABB LOSO harness** (Protocol A) on **BNCI2014_001 (2a)** + **BNCI2014_004 (2b)** with **EEGNetv4 + ShallowFBCSPNet** carriers: ERM, `lpc_prior`, `marginal`, `lpc_uniform`, `chain`. Use offline cache (`configure_offline_moabb`). *(core result table)*
3. **Frozen-encoder leakage metric** (reuse `leakage_probe`) reported alongside accuracy for every method.
4. **Tier-1 baselines: CDANN, EEG-DG, IIB** on the same LOSO splits. CDANN vs `lpc_prior` is the headline experiment.
5. **Cross-session (Protocol B)** on 2a/2b (`D`=subject×session).

### SHOULD-HAVE (strengthens the paper)
6. **Architecture-agnosticism:** repeat `lpc_prior` vs ERM with **EEGConformer** (`return_features`) and **ATCNet**; **Deep4Net** capacity-stress ablation.
7. **Tier-2 baselines:** DANN, CORAL/MMD, C-CORAL/C-MMD, VREx, GroupDRO via DomainBed penalties.
8. **Scale-up LOSO:** Lee2019_MI (OpenBMI, 54 subj) — large-scale credibility.
9. **CIRCE** as the kernel-conditional competitor/alternative carrier.

### OPTIONAL (if time permits)
10. **Cross-dataset binary MI (Protocol C):** 2b ↔ OpenBMI ↔ Cho2017 (left vs right, channel intersection).
11. **TSMNet/SPDDSMBN** geometric baseline; **LogCov+TS** frozen-carrier ablation.
12. **SCLDGN SupCon** contrastive baseline; conditional-InfoNCE lower-bound bracket.
13. **EEG foundation-model carriers** (LaBraM/CBraMod/EEGPT) — *separate braindecode>=1.4 env only*; pure frozen-feature leakage probe. Explicitly out of the AAAI core given the pinned 0.8 stack.

**Efficiency-first ordering rationale:** items 1–5 reuse already-validated code (`sanity_check.py`) and the two smallest datasets (2a/2b, 9 subjects each) → fastest path to a complete, defensible result on V100/A100. Architecture-agnosticism (6) and the baseline breadth (7) are cheap add-ons because the harness is encoder-agnostic and DomainBed penalties are ~15-line drop-ins. Cross-dataset and foundation models are the expensive long-pole items deferred to optional.

---

## 7. Key file references
- `/home/infres/yinwang/CMI_AAAI/synthetic/sanity_check.py` — **the estimator + 4 ablations + leakage probe, already validated** (reuse `kl_to_prior`, `empirical_priors`, `train_one` alternation, `leakage_probe`).
- `/home/infres/yinwang/CMI_AAAI/synthetic/results.json` — Protocol-D results.
- `/home/infres/yinwang/CMI_AAAI/cmi/paths.py` — offline MOABB config (`configure_offline_moabb`), MI dataset registry, datalake path `/projects/EEG-foundation-model/datalake/raw`.
- `/home/infres/yinwang/CMI_AAAI/README.md` — protocols, datasets, metrics, env (authoritative).
- `/home/infres/yinwang/CMI_AAAI/scripts/slurm_template.sh` — SLURM submission (login node has no GPU; all training via `sbatch`, V100/A100).