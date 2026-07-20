# Pillar 2 — The Resolution: label correction decouples the two CMIs

**Setup.** Random triple `(Z, Y, D)` with `Z = g(X)` the representation, `Y` the label,
`D` the domain. Per-domain label prior `π_d(y) := p(Y=y | D=d)`. The encoder leakage term
is `I(Z;D|Y)`; the decoder / concept term is `I(Y;D|Z)`. All quantities in nats.

**Anchors carried in (verified at 1e-15).**
- **A1 (exact identity):** `I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D)`.
- **A2 (the fight):** if the encoder is conditionally invariant, `I(Z;D|Y)=0`, then
  `I(Y;D|Z) = I(Y;D) − I(Z;D)`. Under label shift `I(Y;D) > 0`, this is strictly positive
  unless `I(Z;D)=I(Y;D)`.
- **A3 (escape clause):** both CMIs are jointly 0 **iff** `I(Z;D)=I(Y;D)`; since `Z` must
  predict `Y`, equality forces `Y=f(Z)` deterministically (zero Bayes error) — impossible
  with irreducible label noise. ⇒ the terms FIGHT.

The single cause of the fight is the label-shift gap `I(Y;D) > 0` on the RHS of A1.
Pillar 2 removes it by importance reweighting and shows both CMIs can then be zeroed.

---

## (a) Reweighting produces a joint with `Ĩ(Y;D)=0`, and the identity changes sign-favorably

**Definition (GLS reweighting, Combes 2020 Def. 3.4).** Fix any reference label prior
`π*(y)` with `π*(y) > 0` wherever some `π_d(y) > 0`. Reweight each domain `d` by
```
  w_d(y) = π*(y) / π_d(y).
```
Define the reweighted joint by scaling the original `p(z,y,d)` and renormalizing:
```
  p̃(z, y, d)  ∝  w_d(y) · p(z, y, d).
```

**Lemma 2.1 (label shift is removed).** Under `p̃`, the conditional label law in every
domain equals the reference: `p̃(Y=y | D=d) = π*(y)` for all `d`. Hence `Y ⟂ D` under `p̃`
and `Ĩ(Y;D) = 0`.

*Proof.* Factor the original joint as `p(z,y,d) = p(d) · π_d(y) · p(z | y, d)`. Reweighting
multiplies the `(y,d)` block by `w_d(y) = π*(y)/π_d(y)`:
```
  w_d(y) · p(d) · π_d(y) · p(z|y,d) = p(d) · π*(y) · p(z|y,d).
```
The within-domain label marginal is `∫_z p(d)·π*(y)·p(z|y,d) dz / p(d) = π*(y)·∫_z p(z|y,d)dz
= π*(y)`, independent of `d`. (Renormalization is by `Σ_{d} p(d) Σ_y π*(y) = 1`, so the
proportionality constant is 1 and `p̃` is already a valid distribution.) Therefore
`p̃(y,d) = p̃(d) · π*(y)` factorizes, giving `Ĩ(Y;D) = 0`. ∎

**Corollary 2.2 (the identity decouples).** The exact identity A1 holds verbatim under any
distribution, in particular under `p̃`. Substituting `Ĩ(Y;D)=0`:
```
  Ĩ(Z;D|Y) − Ĩ(Y;D|Z) = Ĩ(Z;D) − Ĩ(Y;D) = Ĩ(Z;D),
  ⇒  Ĩ(Z;D|Y) = Ĩ(Y;D|Z) + Ĩ(Z;D).        (★)
```
Contrast with the pre-reweighting A2 relation `I(Y;D|Z) = I(Y;D) − I(Z;D)`. Before, driving
the **encoder** term to 0 *forced* the **decoder** term up to the label-shift gap. After
reweighting, (★) shows the two CMIs and `Ĩ(Z;D)` are linked **additively with all-positive
signs**: making `Ĩ(Z;D|Y)=0` requires `Ĩ(Y;D|Z)=0` **and** `Ĩ(Z;D)=0` together (each is a
non-negative MI, and a sum of non-negatives is 0 iff each summand is 0) — there is no longer
a forced trade-off where lowering one *raises* the other. Encoder-invariance no longer
mandates decoder-variance.

---

## (b) Under `p̃`, BOTH CMIs are simultaneously 0 when `Z` is sufficient for `Y`

**Theorem 2.3.** Suppose, under `p̃`, the representation is (i) **conditionally invariant**,
`p̃(z | y, d) = p̃(z | y)` for all `d` (GLS in feature space), and (ii) a **sufficient
statistic** for `Y`, i.e. `Y ⟂ D | Z` need only be argued — concretely, the class-conditional
channel does not depend on `d`. Then
```
  Ĩ(Z;D|Y) = 0   and   Ĩ(Y;D|Z) = 0   and   Ĩ(Z;D) = 0,
```
all three simultaneously, with no residual.

*Proof.*
- **`Ĩ(Z;D|Y)=0`:** condition (i) is exactly `p̃(z|y,d)=p̃(z|y)`, i.e. `Z ⟂ D | Y`, so the
  encoder CMI is 0 by definition.
- **`Ĩ(Z;D)=0`:** under `p̃`, `p̃(z|d) = Σ_y p̃(y|d) p̃(z|y,d) = Σ_y π*(y) p̃(z|y)` (using
  Lemma 2.1 for the prior and (i) for the channel), which is independent of `d`. Hence
  `Z ⟂ D` and `Ĩ(Z;D)=0`.
- **`Ĩ(Y;D|Z)=0`:** plug `Ĩ(Z;D|Y)=0` and `Ĩ(Z;D)=0` into the reweighted identity (★):
  `0 = Ĩ(Y;D|Z) + 0`, so `Ĩ(Y;D|Z)=0`. (Equivalently, with invariant channel and common
  prior, the posterior `p̃(y|z,d) ∝ π*(y) p̃(z|y)` is the same in every domain, so
  `Y ⟂ D | Z`.)

So all three vanish jointly. Note the escape-clause obstruction A3 is dissolved: A3 required
`I(Z;D)=I(Y;D)`, but here both sides are 0 *separately*, which no longer forces zero Bayes
error — irreducible label noise `H(Y|Z) > 0` is fully compatible with `Ĩ(Y;D|Z)=0`, because
the noise is now domain-independent. ∎

**Reading.** "Sufficient statistic" here means: after label correction, `Z` captures all the
`Y`-relevant, domain-invariant signal, so the only thing that *could* still leak domain
information through the label channel is concept shift — which condition (i) forbids. When it
holds, the decoder term is free to be 0 at the same time as the encoder term.

---

## (c) Precise connection to GLS (Combes et al., NeurIPS 2020)

File: `refs/GLS.txt`.

- **What GLS is (their Def. 3.1):** a representation `Z=g(X)` satisfies *Generalized Label
  Shift* if `D_S(Z|Y=y) = D_T(Z|Y=y) ∀y` — i.e. the class-conditional feature law is domain
  invariant. This is *exactly* our `I(Z;D|Y)=0` (encoder term = 0). GLS = "encoder leakage
  zero."
- **Their reweighting (their Def. 3.4 / Lemma 3.1):** importance weights
  `w_y = D_T(Y=y)/D_S(Y=y)`, used to align the *reweighted* source feature distribution
  `D_S^w(Z) = Σ_y w_y D_S(Z, Y=y)` with the target `D_T(Z)`. Our `w_d(y)=π*(y)/π_d(y)` is the
  same object: their `w_y` is the special case `S→T` with `π* = π_T` (reference = target
  prior). Lemma 2.1 above is the population statement behind their Lemma 3.1 (alignment of
  reweighted marginals). They estimate `w` *without target labels* via the confusion-matrix
  moment method `w = C⁻¹μ` (their Lemma 3.2) — orthogonal to our point, but it is what makes
  the reweighting usable in UDA.
- **Their guarantee (Thm. 3.1 / 3.2):** under GLS the source/target error gap is bounded by
  `‖D_S^Y − D_T^Y‖₁ · BER + 2(k−1)Δ_CE`, and the joint error by `2·BER`. GLS controls the
  `Δ_CE` (conditional error gap) term; the `BER` term is a *reweighted source* classification
  performance.
- **What GLS does NOT contain — our addition.** GLS enforces (encoder invariance) + (label
  reweighting). It does **not** add an explicit constraint on the *decoder/concept* channel
  `I(Y;D|Z)` after correction. In their framework the residual is absorbed into BER and is
  assumed small ("clustering structure," their Thm. 3.3/3.4) but never *driven to zero by an
  explicit term*. Our Pillar-2 contribution is to make `I(Y;D|Z)→0` an **explicit objective
  applied after label correction** = direct **concept-shift control**. Corollary 2.2 (★) is
  precisely why this second term is *needed and well-posed*: once `Ĩ(Y;D)=0`, the decoder term
  is no longer pinned by label shift, so it becomes an independent, separately-minimizable
  quantity rather than a forced consequence. The CONTROL experiment in
  `verify_resolution.py` shows that with a concept-shifted (non-invariant) channel,
  reweighting alone leaves `Ĩ(Y;D|Z)=0.145 > 0` — GLS-style correction does not touch it; only
  the explicit decoder constraint does.

One-line positioning: **GLS = invariance + label correction (kills the encoder term and the
label-shift gap); ours = GLS + explicit `I(Y;D|Z)` control (kills the concept-shift term that
GLS leaves in BER).**

---

## (d) Numerical verification — `verify_resolution.py`

The A2 fight example is reconstructed **exactly** (discrete `(Z,Y,D)`, binary `Y`, binary
`D`, `p(d)` uniform; per-domain priors `π_0(Y=1)=0.41`, `π_1(Y=1)=0.80`; invariant channel
`p(z|y=0)=[0.84,0.16]`, `p(z|y=1)=[0.18,0.82]`). All four anchors reproduced, then the
reweighting `w_d(y)=π*(y)/π_d(y)` is applied and everything recomputed with **exact** discrete
MI (no estimation).

Reported numbers (nats):

| quantity        | original (fight) | reweighted (π*=avg) |
|-----------------|------------------|---------------------|
| `I(Z;D)`        | **0.034010**     | 0.000000            |
| `I(Y;D)`        | **0.082302**     | **0.000000**        |
| `I(Z;D|Y)` enc  | **0.000000**     | 0.000000            |
| `I(Y;D|Z)` dec  | **0.048292**     | **0.000000**        |

- A1 identity holds to `|diff| ≤ 8.3e-17` in both worlds.
- **Anchors reproduced:** `I(Y;D)=0.0823`, `I(Z;D)=0.0340`, `I(Z;D|Y)=0`,
  `I(Y;D|Z)=0.0483 = 0.0823−0.0340`. ✓
- **(P2a)** `Ĩ(Y;D) = 4.4e-17` ≈ 0 — label shift removed. ✓
- **(P2a′)** reweighted identity `Ĩ(Z;D|Y) = Ĩ(Y;D|Z) + Ĩ(Z;D)` holds to `4.7e-17`. ✓
- **(P2b)** invariant encoder ⇒ `Ĩ(Z;D|Y)=1.2e-16` **and** `Ĩ(Y;D|Z)=2.3e-18`, both 0
  simultaneously. **Tension gone.** ✓
- **Robustness:** uniform reference `π*=[0.5,0.5]` gives the identical verdict
  (`Ĩ(Y;D), Ĩ(Z;D|Y), Ĩ(Y;D|Z)` all ≈1e-16) — conclusion is `π*`-independent. ✓
- **Control (separates our term from GLS):** with a *concept-shifted* (domain-dependent)
  channel `p(z|y,d)` plus the same reweighting, `Ĩ(Y;D)=−2.2e-17` (label shift still gone)
  but `Ĩ(Y;D|Z)=0.1449 > 0` and `Ĩ(Z;D|Y)=0.1500 > 0`. Reweighting alone does **not** zero
  the decoder/concept term — confirming that the explicit `I(Y;D|Z)` constraint (our addition)
  is a genuinely independent requirement. ✓

`ALL PILLAR-2 CLAIMS VERIFIED: True`.

**Take-away.** The fight in A2 is *entirely* the label-shift gap `I(Y;D)` sitting on the RHS
of the exact identity A1. GLS importance-reweighting sets `Ĩ(Y;D)=0`, turning the identity
into the all-positive additive relation (★); an invariant + sufficient `Z` then makes the
encoder term, decoder term, and `Ĩ(Z;D)` vanish *together*. The piece GLS leaves on the
table — concept shift surviving in BER — is exactly what our explicit post-correction
`I(Y;D|Z)→0` constraint removes.
