# H²-CMI — corrected theory (P0-2 … P0-5)

This file records the four theory corrections the review raised, the corrected statements,
and where each is implemented. The H²-CMI package was written so that **none of the
disputed claims appear in code**; the corrected forms are what the modules compute.

---

## P0-2 — the CMI regularizer is a *neural conditional-entropy* estimator, not a posterior-KL upper bound

**Wrong (old AAAI code).** `R(θ,ψ) = E KL(q_ψ(d|z,y) ‖ p(d|y))` was called a variational
*upper bound* on `I(Z;D|Y)`. It is not: setting `q_ψ(d|z,y)=p(d|y)` gives `R=0` while the
true `I(Z;D|Y)` can be strictly positive. `R` equals the true CMI only at
`q_ψ = p_θ(d|z,y)`, so it is a **posterior plug-in**, and even then the Step-A→Step-B
partial gradient is not the profile-CMI gradient.

**Corrected (this package).** Use the identity

```
I(Z; D | Y) = H(D | Y) − H(D | Z, Y)
```

`H(D|Y)` is encoder-independent and estimated empirically; `H(D|Z,Y)` is the *optimal*
conditional cross-entropy of a neural critic `q_ψ(d|z,y)`. The encoder **maximises** that
optimal conditional cross-entropy (a min–max / conditional-GRL objective). At the critic
optimum `ψ*(θ)` the envelope theorem makes the encoder gradient equal the partial
derivative holding `ψ` fixed — so freezing the critic in Step B and back-propagating
`−λ·∂CE/∂z` is the correct profile gradient.

Code: [`cmi/hierarchical.py`](cmi/hierarchical.py) — `reference_conditional_entropy`
(the `H(D|Y,Pa)` side), `ConditionalDomainCritic` (the `q_ψ`), `HierarchicalCMI.estimate`
returns the **signed** `Î_j = H_ref_j − CE_j`. The penalty added to the loss is
`Σ_j λ_j·Î_j` with `λ_j ≥ 0`; minimising it maximises `CE_j` (reduces leakage). The
reporting estimator ([`eval/leakage.py`](eval/leakage.py)) is **cross-fitted** and ships a
within-`(Y,Pa)` permutation null; negatives are kept (no truncation → no upward bias).

---

## P0-3 — there is no "joint CMI = 0 ⇔ zero Bayes error" theorem

**Wrong.** Under `D→Y→Z`, the claim was that `I(Z;D|Y)=0` and `I(Y;D|Z)=0` simultaneously
hold **iff** `Y=f(Z)` (zero Bayes error).

**Counterexample (kept in mind, not asserted anywhere).** `A,B` independent; `D=A`,
`Y=(A,B)`, `Z=A`. Then `D→Y→Z` holds and `I(Z;D|Y)=I(Y;D|Z)=0`, yet `H(Y|Z)=H(B)>0`, so
`Y` is **not** determined by `Z`. The DPI equality condition is `D ⊥ Y | Z` (Z sufficient
for the `D`-information carried by `Y`), **not** full recovery of `Y` from `Z`.

**Corrected.** We keep only what is true: the chain-rule coupling
`I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D)`; that label-proportion differences create tension
between marginal and class-conditional invariance; and that stronger conclusions need
extra assumptions (binary task, minimal-sufficient representation). H²-CMI therefore does
**not** use a residual decoder `p(y|z,d)` as a training objective; the decoder residual is
a **diagnostic** only (and feeds the safety gate). See P0-4.

---

## P0-4 — `I(Y;D|Z)` is a *conditional predictive-insufficiency* diagnostic, not "genuine concept shift"

`I(Y;D|Z) = H(Y|Z) − H(Y|Z,D)` is the residual gain from knowing the domain after the
representation. It can arise from a true `p(y|x,d)` change **or** from `Z` discarding
task-relevant information, classifier misspecification, incomplete label-shift correction,
domain-specific calibration/annotation noise, or thin class support per domain. It is only
*evidence* of concept shift under correct reweighting, sufficient `Z`, a flexible
predictor and good overlap.

**The clinical degeneracy.** If each subject's disease label is fixed (`D=subject ⇒
Y=g(D)`), then `I(Y;D|Z)=H(Y|Z)` and the "concept" probe degenerates to label
predictability.

**Corrected (this package).**
* The domain DAG marks this explicitly: `DomainFactor.determines_label` and
  `DomainDAG.hierarchical_site_subject_session(subject_determines_label=…)`
  ([`domains/dag.py`](domains/dag.py)).
* Encoder invariance is applied to **acquisition** factors (site/montage/session), never to
  a label-mechanism factor; rater/site **label** effects are modelled by the latent-`Y*`
  mechanism ([`label/site_mechanism.py`](label/site_mechanism.py)), not by encoder
  invariance.
* The concept/insufficiency residual is used to **gate** adaptation, not as an encoder
  penalty.

---

## P0-5 — the GLS reference domain distribution

**Wrong.** With GLS weights `w(d,y)=π*(y)/p(y|d)`, the reference domain marginal was coded
as `p_d_ref = pi_y.mean(0)` (i.e. `mean_y p(d|y)`). In general
`(1/|Y|)Σ_y p(d|y) ≠ p(d)`, so training target, the GLS-induced measure, and the
evaluation reference were three different probability measures.

**Corrected.** The reweighted dataset induces

```
p_ref(d) = normalise( Σ_i  w_i · 1[d_i = d] )  with  w_i = π*(y_i)/p(y_i|d_i)
```

i.e. a **bincount of `d` weighted by the GLS weights** (≈ `p(d)` under fully within-domain
GLS). Code: [`align/reference_marginal.py`](align/reference_marginal.py) —
`gls_weights`, `gls_reference_domain_marginal`.

We also fix the related modelling error (review §4): forcing **raw** marginals
`p_d(z)` equal is wrong under label shift, because `p_d(z)=Σ_y p_d(y)p_d(z|y)` differs
across domains whenever `p_d(y)` differs. H²-CMI aligns the marginal under a **fixed
reference prior**

```
p_d*(z) = Σ_y π*(y) p_d(z|y)
```

realised by aligning each domain's **class-conditional** `p_d(z|y)` to the pooled
`p(z|y)`, weighted by `π*(y)` (`ReferenceMarginalAlignment`). This never collapses class
structure under label shift and is consistent with the shared class-conditional density
head `p_φ(z|y)`.

---

## What the method then is (one sentence)

> An explicit domain-factor **DAG**, controlled by an exact **hierarchical conditional-MI
> decomposition** with per-factor **leakage budgets**; a shared **class-conditional latent
> density** `p_φ(z_c|y)` learned under a fixed reference prior; test-time **EEG-constrained
> low-dimensional canonicalisation + target-prior estimation** at the source class-conditional
> geometry; and a **source-only learned safety gate** that executes or refuses adaptation.

The package additionally encodes the three negative boundaries the review established, so
they are respected by construction rather than bypassed by more penalties:

1. invariance does not automatically raise accuracy → leakage is **budgeted**, not
   maximised, and accuracy/TTA are evaluated separately ([`eval/harness.py`](eval/harness.py));
2. pooled alignment harms under label/concept shift → alignment is **reference-prior
   class-conditional**, and TTA is **gated** by a learned harm predictor;
3. single-class targets are unidentifiable for class-conditional transport → TTA **falls
   back to identity** below `min_effective_classes`/`min_target`
   ([`tta/class_conditional.py`](tta/class_conditional.py)).
