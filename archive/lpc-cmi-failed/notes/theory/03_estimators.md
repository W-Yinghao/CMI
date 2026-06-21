# 03 — Variational estimators (what we actually optimize)

Builds on the verified anchors (do not contradict):
- **A1 (exact identity):** `I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D)`.
- **A2 (tension):** if `I(Z;D|Y)=0` then `I(Y;D|Z) = I(Y;D) − I(Z;D)`; strictly `>0` under label shift unless `I(Z;D)=I(Y;D)`.
- **A3 (escape):** both CMIs vanish jointly **iff** `I(Z;D)=I(Y;D)`, i.e. `Y=f(Z)` deterministically — impossible under clinical-EEG label noise, so the two terms **fight**.
- **A4 (resolution = GLS):** importance-reweight domain `d` by `w_d(y)=π*(y)/π_d(y)` to a common reference prior `π*`; in the reweighted law `Ĩ(Y;D)=0`, which **decouples** the CMIs so both can be driven to 0.

We never observe the true MIs; we optimize **two tractable variational surrogates** — one per CMI — plus the task loss. Notation: `Z=g(X)` encoder features, `Y` label, `D` domain (subject/cohort), `π_y(D)=p(D|Y=y)` the per-class domain prior.

---

## (a) The two estimators and their bias directions

### Encoder term — `I(Z;D|Y)` via posterior-KL (UPPER bound)

Exact CMI:
```
I(Z;D|Y) = E_{z,y} KL( p(D|z,y) || p(D|y) ) = E_{z,y} KL( p(D|z,y) || π_y(D) ).
```
Replace the intractable true posterior `p(D|z,y)` with a **variational classifier** `q_ψ(D|z,y)` (an MLP). Using the Barber–Agakov decomposition, for **any** `q_ψ`:
```
E_{z,y} KL( q_ψ(D|z,y) || π_y(D) )
  = I(Z;D|Y) + E_{z,y} KL( p(D|z,y) || q_ψ(D|z,y) )     [the second term ≥ 0]
  ≥ I(Z;D|Y).                                            (★ UPPER bound)
```
So `L_enc := E KL(q_ψ(D|Z,Y) || π_y(D))` is an **upper bound**, tight iff `q_ψ = p(D|z,y)`.
This is exactly the code's `kl_to_prior(q_dzy(...), log_pi_y[y])` — see (c).

- **Counted vs variational:** `π_y(D)` is the **counted/plug-in** prior (`empirical_priors`/`subject_priors`/`effective_priors`); only `q_ψ` is learned. Putting the learned object in the *first* slot of the KL (numerator) is what makes the bound upper, not lower — it's the encoder-facing direction we want to *minimize*.
- **Bias direction:** **over-estimates** `I(Z;D|Y)` whenever `q_ψ` is sub-optimal (Step-A under-fit). Minimizing an upper bound is **safe**: driving `L_enc→0` forces the true CMI to 0. The danger is the opposite — a *weak* `q_ψ` reports a small `L_enc` only if it genuinely can't read `D` off `(Z,Y)`; a lazy `q_ψ` would *inflate* the bound, not hide leakage. Hence Step-A must be run to (near) convergence (`n_inner` inner steps).
- **Numpy check (this repo, `np` exact):** with a discrete joint, optimal `q_ψ` gives `E KL = I(Z;D|Y) = 0.194167` exactly; injecting noise into `q_ψ` yields `0.236, 0.412, 0.689, …` — **always ≥** the exact value. Confirms (★).

### Decoder term — `I(Y;D|Z)` via entropy difference (consistent plug-in; one-sided pieces)

Identity:
```
I(Y;D|Z) = H(Y|Z) − H(Y|Z,D).
```
Estimate each conditional entropy by the **cross-entropy of a fitted predictor** (this is the IIB construction):
```
H(Y|Z)    ≈ CE_q  := E[ −log q(Y|Z) ]      ,  q = task head p(y|z)  (here: the classifier logits)
H(Y|Z,D)  ≈ CE_h  := E[ −log h(Y|Z,D) ]    ,  h = q_dzy's sibling h(Y|Z,D) MLP
  ⇒  Î(Y;D|Z) := CE_q − CE_h.
```
Bound bookkeeping — **each cross-entropy is itself an upper bound on its entropy** (Gibbs):
```
CE_q ≥ H(Y|Z)      (equality iff q = p(y|z))
CE_h ≥ H(Y|Z,D)    (equality iff h = p(y|z,d)).
```
So the **difference is two-sided** and *not* a clean single-sided bound in general:
```
Î(Y;D|Z) = CE_q − CE_h = I(Y;D|Z) + [CE_q − H(Y|Z)] − [CE_h − H(Y|Z,D)].
                                     └── q gap ≥0 ──┘   └── h gap ≥0 ──┘
```
- If `q` is the **same task head** the encoder is already pushing to minimize CE_q, and `h` is fit to optimality in Step-A, then the `q`-gap ≥ 0 and `h`-gap = 0, giving `Î(Y;D|Z) ≥ I(Y;D|Z)` (**over-estimate**, conservative for the decoder penalty).
- If instead `q` is optimal but **`h` is under-fit** (Step-A too short), the `h`-gap > 0 and the estimate is **pulled down** → **under-estimate** (lower bound). My numpy check: with optimal `q,h`, `CE_q−CE_h = H(Y|Z)−H(Y|Z,D) = 0.198942 = I(Y;D|Z)` exactly; with a *sub-optimal* `h`, `CE_h` rises `0.474→0.542` and the estimate drops `0.199→0.132 ≤` exact — confirming the lower-bound regime when `h` lags.
- **Net:** `Î(Y;D|Z)` is a **consistent plug-in** (exact at the joint optimum `q=p(y|z), h=p(y|z,d)`); its finite-fit bias is **governed by the relative fit of `q` vs `h`**. Practically `h` sees strictly more information (`Z,D`) than `q` (`Z`), so once both are well-fit `CE_h ≤ CE_q` and the estimate is non-negative, as it must be.

**Why two different machineries.** The encoder term needs only a *domain* classifier + a *fixed counted prior* → a pure KL we can make an honest upper bound. The decoder term is about *label predictability* and is naturally an entropy gap, so it reuses the task head (`q`) and an auxiliary domain-conditioned head (`h`). This asymmetry is exactly anchor A1's asymmetry: `I(Z;D|Y)` lives on the *encoder* side (invariant `p(z|y)`), `I(Y;D|Z)` on the *decoder* side (invariant `p(y|z)`).

---

## (b) Joint objective and two-step optimization

### Objective
```
L  =  CE_balanced(Y, ŷ)                       # task loss, optionally class-reweighted (label correction)
      + λ_enc · I(Z;D|Y)                       # encoder leakage, posterior-KL UPPER bound
      + λ_dec · I(Y;D|Z)                       # decoder leakage, entropy-gap plug-in
```
with the estimators of (a):
```
I(Z;D|Y) ≈ E_{z,y} KL( q_ψ(D|z,y) || π_y(D) )
I(Y;D|Z) ≈ CE_q(Y|Z) − CE_h(Y|Z,D).
```
**Label correction (GLS / anchor A4).** Two interchangeable knobs make the task loss target the *balanced* / reweighted risk so the two CMIs can be *jointly* zeroed (A3 escape is otherwise blocked under label shift):
1. **Class-reweighted CE** — `ce_weight ∝ 1/freq(y)` (the `balance=True` path, see (c)); this is the Balanced-Error-Rate / GLS reweighting to a uniform reference `π*`.
2. **Prior choice for `π_y`** — `empirical` vs `effective`(uniform-over-present) vs `subject`; the prior must be *consistent with the sampler* (else the KL target is mis-specified). `effective_priors` is the in-objective realization of reweighting domains to uniform within a class.

### Two-step alternating optimization
Per minibatch (and `n_inner` inner repeats for Step A):

- **Step A — fit the posteriors / auxiliary heads on _detached_ Z** (no encoder gradient):
  ```
  z = g(x).detach()
  min_ψ  CE( q_ψ(D|z,y), d )           # tighten the encoder UPPER bound
  min_h  CE( h(Y|z,d),   y )           # tighten H(Y|Z,D) estimate for the decoder term
  ```
  Detaching is essential: Step A must *maximally* read `D` from `(Z,Y)` to make the bound honest; if the encoder could move here it would cheat by hiding `D` from a deliberately-weak `q`.

- **Step B — update encoder + task head with Z carrying gradient**, posteriors frozen:
  ```
  z = g(x)                              # grad flows
  L = CE_bal + λ_enc · KL(q_ψ(D|z,y) || π_y[y]) + λ_dec · (CE_q − CE_h(z,y,d))
  ```
  Here `q_ψ, h` are evaluated but **not updated**; only `g` (and task head) move, pushing `Z` to (i) make `D` unreadable given `Y` and (ii) make `Y` no-more-predictable from adding `D`.

- **Warm-up:** `λ_t = λ · min(1, ep/warmup)` (and `γ_t` likewise) ramps the CMI penalties so the encoder first learns a usable representation before invariance pressure — prevents the degenerate `Z⊥everything` collapse early on.

This is the standard min–max-free alternating scheme (it is *not* adversarial: Step A minimizes CE, Step B minimizes a bound w.r.t. a different parameter set), which is why it's more stable than DANN-style GRL.

---

## (c) EXACTLY what the implemented `dual` method and `balance` flag do (with code citations)

### The `balance` flag — `cmi/train/trainer.py:67–71`
```python
ce_weight = None
if balance:
    cnt = np.bincount(ytr, minlength=n_cls).astype("float32")
    ce_weight = torch.tensor((cnt.sum() / (n_cls * np.maximum(cnt, 1))), dtype=torch.float32, device=device)
```
- Builds a **per-class CE weight `= N / (n_cls · count_y)`**, i.e. inverse-frequency normalized so weights average to 1. This is the **Balanced-Error-Rate / GLS label-shift correction** (anchor A4): it reweights the *task* loss to a uniform class reference `π*(y)=1/n_cls`.
- It is consumed **only** in the task CE: `ce_q = F.cross_entropy(logits, yb, weight=ce_weight)` (`trainer.py:198`). When `balance=False`, `ce_weight=None` → ordinary (natural-frequency) CE.
- The comment at `:67` names it precisely: *"class-balanced (BER) CE weights — the label-shift correction (GLS): each class weighted by 1/freq."*
- **Scope:** it does **not** touch the CMI estimators or the priors `π_y`; it is purely the task-side leg of the label correction in the objective `L` of part (b). (The sampler `classbal`/`domainbal` and `prior_mode` are the *other*, orthogonal correction knobs — `trainer.py:22–36`, `:48–51`.)

### The `dual` method — joint encoder + decoder invariance
`is_dual = method == "dual"` (`trainer.py:66`). It is the **only** method that optimizes *both* CMIs simultaneously — the direct implementation of the A1/A2 two-term picture.

**Step A (fit both posteriors on detached Z)** — `trainer.py:172–182`:
```python
if uses_cmi or is_iib or is_dual:
    with torch.no_grad():
        _, z = backbone(xb)
    for _ in range(n_inner):
        if is_dual:                          # fit q(D|Z,Y) [encoder] AND h(Y|Z,D) [decoder]
            la = post.posterior_loss(z, yb, db) + post.iib_ce_h(z, yb, db)
        ...
        opt_post.zero_grad(); la.backward(); opt_post.step()
```
- `post.posterior_loss` (`regularizers.py:87–93`) is the Step-A CE that fits `q(D|Z,Y)` **and** `q(D|Z)`, `q(S|Z)`; for `dual` only `q_dzy` is used downstream.
- `post.iib_ce_h` (`regularizers.py:95–99`) fits the auxiliary **`h(Y|Z,D)`** predictor (`h_ydz`, an MLP on `[z, onehot(d)]`).
- So `dual` is the **union** of the CMI posterior machinery (encoder) and the IIB auxiliary head (decoder), both fit on the same detached `z`.

**Step B (encoder update, both penalties)** — `trainer.py:206–211`:
```python
if is_dual:                              # DUAL: encoder I(Z;D|Y) + decoder I(Y;D|Z)
    r_enc = post.reg("lpc_prior", z, yb)             # I(Z;D|Y): invariant p(z|y)
    r_dec = ce_q - post.iib_ce_h(z, yb, db)          # I(Y;D|Z): invariant p(y|z) = H(Y|Z)-H(Y|Z,D)
    loss = loss + lam_t * r_enc + gamma_t * r_dec
```
- **`r_enc`** = `post.reg("lpc_prior", z, y)` → `kl_to_prior(q_dzy([z,onehot(y)]), log_pi_y[y])` (`regularizers.py:113–114`). This is **exactly `L_enc`** of part (a): the posterior-KL **upper bound** on `I(Z;D|Y)` against the **counted per-class prior `π_y`**.
- **`r_dec`** = `ce_q − post.iib_ce_h(z,y,d)` = `CE_q(Y|Z) − CE_h(Y|Z,D)` = **`Î(Y;D|Z)`** of part (a) (the entropy-gap plug-in). `ce_q` is the **same task-head CE** already in `loss` (so `q` is shared, per the (a) bookkeeping); `iib_ce_h` reuses the Step-A-fitted `h` but now **with gradient flowing into `z`**.
- **Weights map directly to the objective:** `λ_enc ≡ lam_t` (warmed `lam`), `λ_dec ≡ gamma_t` (warmed `gamma`). Warm-ups: `lam_t` `trainer.py:118`, `gamma_t` `trainer.py:119`.
- **Sign/role check vs anchors:** `r_enc` enforces invariant `p(z|y)` (drives `I(Z;D|Y)→0`, the A2 premise); `r_dec` simultaneously penalizes residual `I(Y;D|Z)` (the term A2 says is otherwise forced `>0` under label shift). `dual` only escapes the A2/A3 tension when paired with the label correction (`balance=True` and/or matched `prior_mode`) — exactly anchor A4.

**Contrast with neighbors (so `dual` is unambiguous):**
- `iib` (`trainer.py:204–205`) optimizes **only** the decoder term `lam_t·(ce_q − iib_ce_h)` — no `r_enc`.
- `lpc_prior` (via `uses_cmi`, `trainer.py:212–214` → `cmi_method`) optimizes **only** the encoder term `lam_t·r_enc` — no `r_dec`.
- `dual` = `lpc_prior`'s `r_enc` **+** `iib`'s `r_dec`, with **independent weights** `lam_t` (encoder) and `gamma_t` (decoder). It is the literal code realization of the A1 decomposition `{I(Z;D|Y), I(Y;D|Z)}`.

---

### One-line summary table

| term | estimator (code) | bound direction | learned vs counted | weight |
|---|---|---|---|---|
| `I(Z;D|Y)` | `KL(q_dzy ‖ log_pi_y[y])` (`reg("lpc_prior")`) | **upper** bound (Barber–Agakov) | `q_ψ` learned, `π_y` counted | `λ_enc=lam_t` |
| `I(Y;D|Z)` | `ce_q − iib_ce_h` (`H(Y|Z)−H(Y|Z,D)`) | consistent plug-in; over-est if `q` loose, under-est if `h` loose | `q`=task head, `h` learned | `λ_dec=gamma_t` |
| task / label corr. | `cross_entropy(logits,y,weight=ce_weight)` | — | `ce_weight=N/(n_cls·count_y)` when `balance=True` | 1 |

Numpy verifications (exact, `/home/infres/yinwang/anaconda3/bin/python3`): encoder `E KL ≥ I(Z;D|Y)` for all perturbed `q` (equality at optimum, 0.194167); decoder `CE_q−CE_h = I(Y;D|Z)` exactly at the optimum (0.198942) and drops below it when `h` is under-fit.
