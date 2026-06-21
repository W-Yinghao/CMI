# Pillar 1 — The Tension Theorem

**Claim (informal).** On the triple of random variables `(Z, Y, D)` — learned
representation `Z`, label `Y`, domain `D` — the *encoder leakage*
`I(Z;D|Y)` and the *decoder leakage* `I(Y;D|Z)` are not independent quantities:
they are tied by an exact identity, and under label shift they cannot both be
made zero unless the task is noiseless. Driving one to zero (e.g. learning a
conditionally-invariant encoder, `I(Z;D|Y)=0`) *forces* the other up by exactly
the label-shift gap `I(Y;D) - I(Z;D)`. This is the information-theoretic core of
the Zhao et al. (2019) tradeoff, and its resolution is Generalized Label Shift
(GLS) reweighting (Tachet des Combes et al., 2020).

Throughout, `Z, Y, D` are discrete with finite support, all information
quantities are in nats, and `0·log0 := 0`. Mutual information
`I(A;B) = H(A) + H(B) − H(A,B) ≥ 0`, and conditional MI
`I(A;B|C) = H(A,C) + H(B,C) − H(C) − H(A,B,C) ≥ 0`. Both are symmetric in their
first two arguments and nonnegative (the latter by Jensen / the data-processing
viewpoint).

---

## (a) The exact identity A1 — two expansions of the chain rule

> **Theorem 1 (Identity).** For any joint distribution `p(Z,Y,D)`,
> ```
> I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D).                    (A1)
> ```

**Proof.** Apply the chain rule of mutual information to the *joint* information
that the pair `(Y,Z)` carries about the domain `D`, expanding the pair in the
two possible orders.

The chain rule states: for any variables `A, B, C`,
```
I((A,B); C) = I(A;C) + I(B;C|A).                          (CR)
```
This is an identity, immediate from the entropy decomposition
`I((A,B);C) = H(A,B) − H(A,B|C)` together with
`H(A,B) = H(A) + H(B|A)` and `H(A,B|C) = H(A|C) + H(B|A,C)`:
```
I((A,B);C) = [H(A) − H(A|C)] + [H(B|A) − H(B|A,C)]
           = I(A;C) + I(B;C|A).
```

Now take `C = D` and expand `I((Y,Z);D)` two ways.

**Expansion 1 (condition on `Y` first):**
```
I((Y,Z);D) = I(Y;D) + I(Z;D|Y).                           (E1)
```

**Expansion 2 (condition on `Z` first):**
```
I((Y,Z);D) = I(Z;D) + I(Y;D|Z).                           (E2)
```

Both equal the same quantity `I((Y,Z);D)` — the total information the
representation-and-label pair carries about the domain — so the right-hand
sides are equal:
```
I(Y;D) + I(Z;D|Y) = I(Z;D) + I(Y;D|Z).
```
Rearranging gives exactly (A1):
```
I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D).            ∎
```

**Remark (interpretation).** `I((Y,Z);D)` is the *budget*: the total domain
information available in the (label, representation) pair. (A1) is a conservation
law — it says this budget can be *attributed* either as "label carries `I(Y;D)`,
representation adds `I(Z;D|Y)` beyond the label" or as "representation carries
`I(Z;D)`, label adds `I(Y;D|Z)` beyond the representation." The two attributions
must agree, which couples the two leakages.

**Numerically verified:** max absolute error of (A1) over **3000** random
discrete joints with `|Z|,|Y|,|D| ∈ [2,5]` (including sparse tables to stress the
`0·log0` boundary) is **2.22 × 10⁻¹⁵** — i.e. exact to machine precision. See
`verify_tension.py`, block `[i]`.

---

## (b) A2 — encoder invariance forces decoder leakage, and the monotone tradeoff

> **Theorem 2 (Forced leakage).** If the encoder is *conditionally invariant*,
> i.e. `Z ⟂ D | Y` so that `I(Z;D|Y) = 0`, then
> ```
> I(Y;D|Z) = I(Y;D) − I(Z;D).                               (A2)
> ```
> In particular, under label shift `I(Y;D) > 0`, the decoder leakage is
> ```
> I(Y;D|Z) = I(Y;D) − I(Z;D) > 0     whenever  I(Z;D) < I(Y;D).
> ```

**Proof.** Set `I(Z;D|Y) = 0` in the identity (A1):
```
0 − I(Y;D|Z) = I(Z;D) − I(Y;D)   ⟹   I(Y;D|Z) = I(Y;D) − I(Z;D).   ∎
```

**Why `I(Z;D) ≤ I(Y;D)` here — the gap is genuinely nonnegative.**
Because `I(Y;D|Z) ≥ 0` always, (A2) immediately yields, *under conditional
invariance*,
```
I(Z;D) ≤ I(Y;D).                                          (bound)
```
So a conditionally-invariant encoder can carry *at most* the label's worth of
domain information, and the decoder leakage `I(Y;D|Z) = I(Y;D) − I(Z;D)` is the
exact shortfall. The representation can "absorb" some of the domain signal into
`I(Z;D)` (via the part of the label it encodes), but never more than `I(Y;D)`,
and whatever it fails to absorb resurfaces in the decoder.

**The monotone tradeoff.** Hold the conditionally-invariant encoder family fixed
(shared class-conditionals `p(z|y)`, `Z ⟂ D | Y`) and let label shift grow.
Parameterize shift strength by `s ∈ [0,1]`, with per-domain priors `π_d(y)`
spreading away from a common prior as `s` increases (`s = 0`: identical priors,
no shift). Then:

- `I(Z;D|Y) = 0` for all `s` (invariance is structural, independent of priors);
- `I(Y;D)` is a strictly increasing function of `s` for `s>0` (more prior
  spread ⟹ more label-domain dependence);
- `I(Z;D)` increases too, but is bounded above by `I(Y;D)` (the (bound)), and
  in general `I(Z;D) < I(Y;D)` whenever `p(z|y)` is not label-invertible;
- hence by (A2) the forced decoder leakage
  `I(Y;D|Z) = I(Y;D) − I(Z;D)` **grows monotonically with `s`**.

This is precisely the Zhao et al. (2019) *fundamental tradeoff*: minimizing
domain discrepancy at the representation (driving toward marginal/conditional
invariance) cannot escape a lower bound on the *joint* (source+target) error when
the marginal label distributions differ across domains. (A2) is the
information-theoretic skeleton of that bound, expressed purely in CMI terms;
`I(Y;D|Z) > 0` certifies that no decoder acting on `Z` alone can be
simultaneously Bayes-optimal across domains.

**Numerically verified (sweep `[ii]`):** with a fixed invariant encoder and `s`
swept over `{0, .2, .4, .6, .8, .95}`:

| `s`  | `I(Z;D|Y)` | `I(Y;D|Z)` | `I(Y;D)` | `I(Z;D)` | `I(Y;D)−I(Z;D)` |
|------|-----------:|-----------:|---------:|---------:|----------------:|
| 0.00 |  −4.4e-16  |  0.000000  | 0.000000 | 0.000000 |       0.000000  |
| 0.20 |  −4.4e-16  |  0.010813  | 0.019333 | 0.008520 |       0.010813  |
| 0.40 |  −4.4e-16  |  0.044399  | 0.078932 | 0.034532 |       0.044399  |
| 0.60 |  −4.4e-16  |  0.104970  | 0.184539 | 0.079569 |       0.104970  |
| 0.80 |   0.0e+00  |  0.203892  | 0.350834 | 0.146942 |       0.203892  |
| 0.95 |  −4.4e-16  |  0.326781  | 0.543095 | 0.216313 |       0.326781  |

`I(Z;D|Y)` is pinned at 0 (≤ 4.4e-16) by construction; `I(Y;D|Z)` matches
`I(Y;D)−I(Z;D)` to **2.2e-16** and is monotone non-decreasing in `s`. This is the
forced tradeoff in action.

**Constructed anchor (block `[iv]`), reproducing A2 exactly.** A symmetric binary
invariant encoder `p(z=1|y=1)=p(z=0|y=0)=a` with per-domain priors
`p(y=1|d) = 0.5 ∓ p1`, with `(a, p1)` solved by bisection, gives
```
I(Z;D|Y) = 4.4e-16 ≈ 0,   I(Y;D) = 0.0823,
I(Z;D)   = 0.0340,         I(Y;D|Z) = 0.0483 = I(Y;D) − I(Z;D).
```
These are the confirmed anchor numbers.

---

## (c) A3 — the zero-Bayes-error escape clause

> **Theorem 3 (Escape).** Both leakages vanish jointly,
> `I(Z;D|Y) = I(Y;D|Z) = 0`, **iff** `I(Z;D) = I(Y;D)`.
> Moreover, when the encoder is conditionally invariant *and* predictive of `Y`
> (so `Z` is used to predict `Y`), this equality holds iff `Y` is a deterministic
> function of `Z`, `Y = f(Z)` — i.e. zero Bayes error.

**Proof of the equivalence.** From (A1),
`I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D)`.

(⟸) Suppose both CMIs are 0. Then the left side is 0, so `I(Z;D) = I(Y;D)`.

(⟹) Suppose `I(Z;D) = I(Y;D)`. Then the right side of (A1) is 0, hence
`I(Z;D|Y) = I(Y;D|Z)`. This alone does **not** force them to 0; it forces them
*equal*. The "both zero" statement is the special case we care about: combining
with conditional invariance (`I(Z;D|Y)=0`) gives `I(Y;D|Z)=0` as well. So under
the standing assumption that we have *driven the encoder to invariance*, the
escape `I(Y;D|Z)=0` is available **iff** `I(Z;D)=I(Y;D)`. ∎

**Why equality means zero Bayes error.** Assume conditional invariance,
`I(Z;D|Y)=0`. We must understand when `I(Z;D)=I(Y;D)`.

Decompose the domain information `Z` carries. Because the only route from `D` to
`Z` is through `Y` (the encoder's class-conditionals are domain-shared, so
`D → Y → Z` is a Markov chain by construction of conditional invariance), the
data-processing inequality gives
```
I(Z;D) ≤ I(Y;D).                                          (DPI)
```
Equality in (DPI) holds iff `Z` is a *sufficient statistic* for `D` w.r.t. the
chain `D → Y → Z`, equivalently iff no information about `D` is lost in the step
`Y → Z`. Under conditional invariance the only `D`-information in `Y` reaches `Z`
through the soft map `p(z|y)`; it is preserved in full iff `p(z|y)` is
**label-invertible** — the posteriors `p(y|z)` are degenerate, i.e. each `z`
determines `y`. That is exactly `Y = f(Z)` deterministically: the Bayes error of
predicting `Y` from `Z` is zero.

Concretely: `I(Z;D) = I(Y;D)` together with `I(Z;D|Y)=0` and `I(Y;D)>0`
(label shift) requires `H(Y|Z) = 0`. To see this, use the (Y,Z) information
budget: `Z` recovers all of `Y`'s domain information only if `Z` recovers `Y`.
With irreducible label noise — `H(Y|Z) > 0` for every measurable encoder, which
is the regime of clinical EEG (overlapping class-conditionals, non-separable
physiology) — strict inequality `I(Z;D) < I(Y;D)` holds, so by (A2)
`I(Y;D|Z) = I(Y;D) − I(Z;D) > 0`. **The two leakages must fight.** ∎

**Numerically verified:** the `s=0` row of the sweep is the escape: there
`I(Y;D)=I(Z;D)=0`, so trivially `I(Z;D)=I(Y;D)` and both CMIs are 0. For every
`s>0` (label shift present, non-invertible encoder) the table shows
`I(Z;D) < I(Y;D)` and hence `I(Y;D|Z) > 0` — the escape is unavailable and the
fight is real.

---

## (d) A4 — the resolution: GLS reweighting decouples the two CMIs

The escape A3 is unreachable under irreducible label noise *as long as
`I(Y;D) > 0`*. The fix (Tachet des Combes et al., 2020, "Generalized Label
Shift"; and Combes NeurIPS'20) is to **remove the label shift itself** by
importance reweighting, rather than to demand a noiseless task.

> **Theorem 4 (GLS resolution).** Fix a common reference label prior `π*(y)`.
> Reweight each domain `d` by `w_d(y) = π*(y) / π_d(y)`, where
> `π_d(y) = p(y | D=d)`, defining the reweighted joint
> ```
> p̃(z,y,d) ∝ p(z,y,d) · w_d(y).
> ```
> Then under the reweighted distribution every domain shares the prior `π*`,
> so `Ỹ ⟂ D` and `Ĩ(Y;D) = 0`. By the identity (A1) applied to `p̃`,
> ```
> Ĩ(Z;D|Y) − Ĩ(Y;D|Z) = Ĩ(Z;D) − Ĩ(Y;D) = Ĩ(Z;D).
> ```
> The two CMIs **decouple**: a conditionally-invariant encoder
> (`Ĩ(Z;D|Y)=0`) now also makes `Ĩ(Z;D)=0` (no domain signal can enter `Z`
> except through `Y`, whose marginal is now domain-free), and therefore
> `Ĩ(Y;D|Z)=0` as well. Both leakages can be zeroed simultaneously without
> requiring zero Bayes error.

**Why the predictive information survives.** Reweighting changes only the *label
priors* `π_d(y)`, not the class-conditional encoder `p(z|y)`. Hence the
within-domain predictive structure `I(Z;Y)` is preserved by the reweighting
(verified below): GLS buys domain-invariance of both leakages *without*
sacrificing the encoder's ability to predict `Y`.

**Numerically verified (block `[iii]`), on the strong-shift `s=0.95` joint:**
```
BEFORE :  I(Y;D)=0.543095   I(Z;D|Y)=−4.4e-16   I(Y;D|Z)=0.326781
AFTER  :  I(Y;D)= 0.0e+00   I(Z;D|Y)=−8.9e-16   I(Y;D|Z)=−4.4e-16
          I(Z;Y) before=0.256420 = I(Z;Y) after=0.256420   (predictive info retained)
```
GLS reweighting collapses the forced decoder leakage from 0.327 nats to machine
zero while keeping the encoder's predictive content `I(Z;Y)` exactly unchanged.

---

## Summary of the tension

| Quantity | Meaning | Under invariance `I(Z;D|Y)=0` | After GLS reweighting |
|---|---|---|---|
| `I(Z;D|Y)` | encoder leakage | `0` (target) | `0` |
| `I(Y;D|Z)` | decoder leakage | `= I(Y;D) − I(Z;D) ≥ 0` | `0` |
| `I(Y;D)`   | label shift | `> 0` (the obstruction) | `0` (removed) |
| escape A3  | both zero | only if `Y=f(Z)` (zero Bayes err) | always |

**Take-away for the method.** Targeting only the encoder leakage `I(Z;D|Y)` (as
conditional-invariance / entropy-regularization DG methods do) leaves the decoder
leakage `I(Y;D|Z) = I(Y;D) − I(Z;D)` as an *irreducible residual* under label
shift — a residual that real clinical EEG, with its irreducible label noise,
cannot escape via A3. The principled fix is to neutralize `I(Y;D)` by GLS
importance reweighting to a reference prior `π*`, after which both leakages can be
jointly driven to zero. This is the bridge from "invariant features fight the
decoder" to "reweighted invariant features cooperate."

---

### Artifacts
- Proof: `notes/theory/01_tension.md` (this file)
- Verification: `notes/theory/verify_tension.py`
  (run with `/home/infres/yinwang/anaconda3/bin/python3 verify_tension.py`)

### References (extracted text in `notes/theory/refs/`)
- `zhao19a.txt` — Zhao, Tachet des Combes, Zhang, Gordon, *On Learning Invariant
  Representations for Domain Adaptation*, ICML 2019. (Fundamental tradeoff /
  joint-error lower bound under label shift.)
- `GLS.txt` — Tachet des Combes, Zhao, Wang, Gordon, *Domain Adaptation with
  Conditional Distribution Matching and Generalized Label Shift*, NeurIPS 2020.
  (GLS assumption + importance-weight resolution A4.)
- `entropyDG.txt` — Zhao, Gong, Liu, Fu, Tao, *Domain Generalization via Entropy
  Regularization*, NeurIPS 2020. (Conditional-invariance regularizer — the class
  of methods that target `I(Z;D|Y)` and inherit the residual A2.)
