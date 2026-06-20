# Tri-CMI unified training objective & the "framework" slots

The carrier is **a standard EEG encoder + task head** (EEGNet/Shallow/Conformer; see
[carrier_design.md](carrier_design.md)). Everything else is a pluggable loss on the
continuous representation `Z = f_θ(X)` plus discrete `Y, D`. The unified objective:

```
L = CE(h_φ(Z), Y)                 # task
  + λ · L_CMI                     # CORE  : LPC-CMI  = E_i KL( q_ψ(D | z_i, y_i) || π_{y_i}(D) )
  + β · Ω(Z)                      # OPT   : compression (dropout/WD/bottleneck, or VIB I(X;Z))
  + γ · L_SupCon(Z, Y, D)         # OPT   : domain-aware supervised contrastive (same-Y, cross-D positives)
```

Training = the CLUB-style two-step alternation already in `synthetic/sanity_check.py`
(Step A: fit `q_ψ` on `Z.detach()`; Step B: encoder+head with task CE + λ·L_CMI + …).

## Regularizer slot — variants (all share q-head machinery; swap to ablate/baseline)
| key | objective | ≈ quantity | role |
|---|---|---|---|
| `lpc_prior` | `E KL(q(D\|Z,Y) ‖ π_y(D))` | `I(Z;D\|Y)` | **CORE (ours)** — conditional domain-leakage, label-prior corrected |
| `lpc_uniform` | `E KL(q(D\|Z,Y) ‖ Uniform)` | — | ablation = the **CDANN target**, mis-specified under imbalance |
| `marginal` | `E KL(q(D\|Z) ‖ p(D))` | `I(Z;D)` | ablation = DANN/CORAL family → **label erasure** under imbalance |
| `chain` | `E KL(q(S\|Z) ‖ p(S))`, S=(D,Y) | `I(Z;D,Y)` | ablation = old super-label → **Y erasure** |
| `iib_swap` | `λ·I(Y;D\|Z)` (diff-of-entropies, 2 heads) | `I(Y;D\|Z)` | ablation = **IIB ordering** (reversed CMI) — see below |
| `circe` | conditional-HSIC kernel penalty | `Z⊥D\|Y` | kernel-side competitor (official MIT) |
| `cond_infonce` | Y-stratified InfoNCE (lower bound) | `I(Z;D\|Y)` | estimator **bracket** (lower bound vs our upper bound) |

## Auxiliary / standalone baselines (separate trainings, same splits)
ERM · DANN · **CDANN** (headline) · Deep CORAL · MMD (+C-CORAL/C-MMD) · VREx · GroupDRO ·
**IIB** · **EEG-DG** · **SCLDGN** (domain-aware SupCon) · TSMNet/SPDDSMBN (geometric).

### Implemented in the harness (`cmi/methods/`, config `method:lam[:gamma]`)
All run on the same backbone via one runner (`cmi/run_loso.py`), shared LOSO splits:
- **posterior-KL CMI** (`regularizers.py`): `erm`, `marginal`, `chain`, `lpc_uniform`, **`lpc_prior`** (ours).
- **contrastive** (`contrastive.py`): `supcon` (domain-aware), **`lpc_supcon`** (hybrid host).
- **DomainBed DG** (`dg_penalties.py`): `coral`, `mmd`, `irm`, `vrex`, `groupdro`, `dann`, **`cdann`**.
- **IIB** (`iib`): core `I(Y;D|Z)=CE_q−CE_h` via auxiliary `h(Y|Z,D)` in `DomainPosteriors`. NOTE: implements
  IIB's *conditional-invariance term* (the reversed-CMI foil); the VIB `I(X;Z)` compression is omitted
  (weight-decay stands in). A full VIB variant (`iib_vib`) is a possible later add.
- ⚠️ Per-framework `lam` is **first-pass** (IRM/VREx need large weights; GroupDRO `lam`=DRO η). A proper
  per-framework lambda sweep is a follow-up before final baseline numbers.

## Contrastive pieces wired in (option (a), done)
- Repos cloned (reference only): [`repos/SCLDGN`](../repos/SCLDGN) (**no license** → reimplement losses)
  ships `lossFunction/{scl,coral,mmd,irm,mcc}.py`; `SupConLoss.forward(features,labels,mask,d=None)`
  is Khosla SupCon + a domain hook. [`repos/SupContrast`](../repos/SupContrast) (**BSD-2** → reusable).
- **Domain-aware SupCon** (`γ·L_SupCon`): positives = same `Y`, *different* `D` (use the `d` arg /
  mask so cross-domain same-class pairs are pulled together) → the geometric counterpart of `I(Z;D|Y)`.
- **Conditional (Y-stratified) InfoNCE**: the lower-bound twin that brackets the LPC-CMI upper bound.

## IIB's roles (NOT the core carrier — see assessment)
IIB objective: `max I(Z;Y) − λ I(Y;D|Z) − β I(X;Z)` with a stochastic (VIB) encoder + an
auxiliary domain-conditioned predictor `h(Y|Z,D)`. Distinction: IIB conditions on **Z** and
constrains the **label–domain** dependence `I(Y;D|Z)` (IRM-style *prediction invariance*);
Tri-CMI conditions on **Y** and removes residual **representation–domain** leakage `I(Z;D|Y)`.
Different CMIs → complementary, not interchangeable.
1. **Headline baseline / theoretical foil** (run full IIB on the same LOSO splits).
2. **Code/template source** (official MIT `Luodian/IIB` on DomainBed): reuse its VIB encoder
   + diff-of-entropies CMI estimation; **borrow its `β·I(X;Z)` VIB term as our optional `Ω(Z)`**.
3. **Swap-in ablation** (`iib_swap`): replace our `I(Z;D|Y)` with IIB's `I(Y;D|Z)` → shows the
   conditioning order matters for EEG leakage.
4. **Optional advanced variant** ("Tri-CMI-VIB", supplement only): `max I(Z;Y) − β I(X;Z) − λ I(Z;D|Y)`
   inside a stochastic encoder. Kept out of the main experiments for optimization stability.
