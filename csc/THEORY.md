# THEORY — Falsifiable Concept-Shift Certificates with Abstention

*When can an unlabeled EEG target tell you the label rule `P(Y|Z)` has changed — and when
must you refuse to answer?*

This file states the framework formally. It is deliberately built **on top of this
project's negative results**, not around them: the A0 falsification
([`notes/A0_FALSIFICATION_FROZEN.md`](../notes/A0_FALSIFICATION_FROZEN.md),
[`notes/A0_SAMPLE_ABSTENTION_PILOT_FROZEN.md`](../notes/A0_SAMPLE_ABSTENTION_PILOT_FROZEN.md))
showed that density / CMI scores are **anti-aligned with adaptation harm**, and the
Evidence Ledger ([`notes/EVIDENCE_LEDGER.md`](../notes/EVIDENCE_LEDGER.md)) names the next
defensible phase as *"identifiability / counterexample theory — a NEW phase, not a seventh
gate score."* That is exactly what this is. We do **not** try to build a universal
concept detector; we characterise the boundary of detectability and **abstain past it**.

---

## 0. Setup and notation

We observe a representation `Z ∈ R^d`, a label `Y ∈ {1..K}`, and a domain index `D`.

* **Source**: multiple *class-spanning* domains with labels, `{(z_i, y_i, d_i)}`. (Real:
  PD medication ON/OFF, or several same-disease cohorts. Synthetic: the simulator.)
* **Target (deployment)**: a batch of `Z` only — **no labels, no source examples** at
  scoring time (the deployment constraint inherited from A0 §1).

A *shift* is a change from the source law of `(Y, Z)` to the target law. We decompose any
shift into a change in the marginal `P(Z)` and a change in the posterior `P(Y|Z)`.

---

## 1. The shift taxonomy (three observable classes + the blind one)

| name | `P(Z)` | `P(Y|Z)` | observable from target `Z` alone? |
|---|---|---|---|
| **covariate** | changes | invariant | **yes** — visible marginal signature |
| **boundary-coupled concept** | changes | changes | **yes** — concept leaves a marginal trace |
| **pure conditional concept** | **invariant** | changes | **NO** — invisible to any `Z`-only detector |

The first two are *support-visible*. The third is the crux.

### 1.1 The impossibility result (why abstention is mandatory)

> **Claim.** Let `f` be any certificate that reads only the unlabeled target marginal
> `P_T(Z)` (no target labels, no fresh source labels at deployment). If two target laws
> `Q_1, Q_2` satisfy `Q_1(Z) = Q_2(Z)` but `Q_1(Y|Z) ≠ Q_2(Y|Z)`, then `f` returns the
> same output on both. Hence no `Z`-only `f` can distinguish "`P(Y|Z)` unchanged" from a
> pure conditional shift. Any `f` that outputs a definite safety verdict here is wrong on
> at least one of `Q_1, Q_2`.

*Proof.* `f` is a function of `P_T(Z)`; `P_T(Z)` is identical under `Q_1, Q_2`; so
`f(Q_1)=f(Q_2)`. ∎

This is the formal version of the A0 finding "you cannot extrapolate a support statistic
into a universal concept detector." The **only** correct output in the regime where the
target marginal is indistinguishable from source is `UNIDENTIFIABLE`. A low density/CMI
reading there is a *false* certificate of safety — precisely the failure A0 documented.

Consequently the certificate is **three-state**, never binary:

```
COVARIATE_ADAPTABLE   visible shift in the covariate atlas; boundary did not move there
CONCEPT_SUSPECT       visible shift aligned with where the source boundary DID move
UNIDENTIFIABLE        abstain: invisible / out-of-atlas / no valid concept atlas
```

---

## 2. Source-side: is there any concept structure to calibrate against?

The certificate can only claim concept evidence if the **source** demonstrates that
`P(Y|Z)` genuinely varies across domains *beyond label shift*. We test this with a
cross-fitted, permutation-calibrated **residual-decoder** statistic.

### 2.1 The statistic

```
T  =  CE( h0(Y | Z, D) )  −  CE( h(Y | Z, D) )
```

* `h0` — **domain-intercept only**: features `[Z, onehot(D)]`. The domain may change the
  per-class prior (label shift) but the `Z→Y` boundary is *shared*.
* `h`  — **domain-dependent boundary**: features `[Z, onehot(D), Z ⊗ onehot(D)]`. The
  interaction lets the boundary move per domain.

`T > 0` (evaluated **out-of-fold**) means: after absorbing per-domain label shift, the
domains still carry domain-dependent *boundary* structure → evidence of genuine concept
variation. Cross-fitting removes the in-sample advantage a richer model would otherwise
get; in practice `T` is a small signed number and we compare it to its null (below).

### 2.2 Permutation calibration

Under the null "no domain-dependent boundary", we permute `D` **within each `Y` stratum**
(destroying `Z|D`-within-class structure while preserving class counts) and recompute `T`.
Significance is `p = (1 + #{T_perm ≥ T}) / (1 + n_perm) ≤ α`. This is the honest threshold;
it accounts for the (negative) cross-fit bias of the richer model.

### 2.3 The support-graph validity gate (the single-class trap)

The residual test is only interpretable as *concept* when the domain–label support graph
is qualified:

* every domain spans `≥ 2` classes (else `D ⇒ Y` and `I(Y;D|Z)` collapses onto label
  predictability `H(Y|Z)` — the **clinical degeneracy** of H2-CMI's P0-4);
* every class appears in `≥ 2` domains (else the boundary is not comparably estimable).

If the gate fails → `status = INVALID` → the certifier **abstains** (no concept atlas can
be built). This is the "single-class subject-domain as an invalid case to reject" from the
proposal, enforced in code.

### 2.4 What `T` is, and is not

`T` is the increment from a *domain-dependent boundary* over a *domain-intercept* decoder.
Per H2-CMI P0-4, the raw `I(Y;D|Z)` is a **predictive-insufficiency diagnostic** that can
fire from `Z` discarding task info, misspecification, incomplete label-shift correction, or
thin support — *not* only from true `p(y|x,d)` change. The intercept-vs-boundary split +
the support gate are exactly what remove the label-shift and degeneracy confounds, leaving
`T` interpretable as boundary movement **conditional on the validity gate**. We make no
claim that `T` is "precise CMI" (the naming correction in the Evidence Ledger is binding).

---

## 3. The source shift atlas

From the source we estimate the *directions* it actually moved along, split to mirror the
`h0`/`h` decomposition. For domain `d`, class `y`, with pooled class mean `μ_y`:

```
a_d      = mean_y ( μ_{d,y} − μ_y )           common (covariate) part: P(Z) moved, boundary did not
r_{d,y}  = (μ_{d,y} − μ_y) − a_d              class-specific residual: the boundary moved
```

* `cov_dirs`     = principal axes of `{a_d}` — where covariate shift is *identifiable*.
* `concept_dirs` = principal axes of `{r_{d,y}}` (with the covariate subspace projected
  out) — where a concept change leaves a *marginal trace*.
* `σ_cov, σ_concept` = the source between-domain spreads (the scale of "normal wobble").

A target shift smaller than that wobble is, by the impossibility result, not
distinguishable from an invisible conditional shift.

---

## 4. The three-state certifier

Given the atlas, the source residual test, and unlabeled target `Z_T`, let
`δ = mean(Z_T) − pooled_mean`, with components `n_cov, n_concept, n_resid` = the norms of
`δ` projected onto `cov_dirs`, `concept_dirs`, and the orthogonal complement, each scaled
by the matching source spread.

```
if source_test.status != VALID:              -> UNIDENTIFIABLE   (no concept atlas)
if max(n_cov,n_concept,n_resid) < τ_detect:  -> UNIDENTIFIABLE   (invisible: pure conditional not excludable)
if n_resid dominates (out of atlas):         -> UNIDENTIFIABLE   (novel direction, identifiability not established)
if n_concept dominates AND source significant:-> CONCEPT_SUSPECT
if n_cov dominates:                          -> COVARIATE_ADAPTABLE
otherwise (ambiguous mix):                   -> UNIDENTIFIABLE
```

The asymmetry is deliberate: the certifier issues `COVARIATE_ADAPTABLE` (a *positive*
safety statement) **only** for a visible shift that lands in the covariate atlas where the
source proved the boundary is stable. Everywhere it cannot see — invisible, out-of-atlas,
ambiguous, or no valid atlas — it abstains. It can be wrong by abstaining too often (low
power); it is designed to never be wrong by **falsely certifying safety**.

---

## 5. What is proven vs assumed (honesty)

* **Proven** (§1.1): a `Z`-only certificate *cannot* identify pure conditional shift;
  abstention there is necessary, not conservative.
* **Calibrated** (§2.2): the source concept-evidence test has a permutation null →
  controllable false-positive rate (a *property of the test*, validated on synthetic).
* **Assumed / to-be-validated**:
  * the linear/mean-shift atlas (§3) captures the relevant geometry. Real EEG concept
    shift may move higher moments or be non-linear in `Z`; the atlas then under-detects
    (more abstention), but the §1.1 guarantee still holds.
  * the thresholds `τ_detect, τ_margin` are currently **hand-set on synthetic** and must
    be calibrated by leave-one-source-domain-out so that a held-out *source* domain is
    rarely mis-certified (this is open work, see PREREGISTRATION §4).
  * "boundary-coupled" concept shift is detectable **only** to the extent the concept
    change perturbs the *observed* marginal; a concept shift that is invisible in `Z` is,
    correctly, `UNIDENTIFIABLE`.

The contribution is therefore not higher average accuracy. It is a **falsifiable formal
boundary**: a certificate that says *when you can diagnose concept shift and when you must
refuse to* — with the refusal proven necessary, and the diagnosis calibrated and
positive-controlled.
