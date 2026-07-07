# Pillar 4 — Positioning & Novelty

Status: built on VERIFIED ANCHORS A1–A4 (exact-MI confirmed to 1e-12; re-checked in this pass).
Scope: position our three-shift CMI decomposition against the three closest prior works.
Honesty mandate: separate what is *genuinely new* from what is *recombination/incremental*.

---

## 0. One-paragraph summary of the claim

We decompose cross-domain risk for representation learning into **three** information-theoretic
shift terms tied to the triple `(Z, Y, D)` (representation, label, domain):

- **covariate shift** = `I(Z; D | Y)`   (encoder term — domain info in Z *after* conditioning on label),
- **label shift**     = corrected by reweighting domain `d` with `w_d(y) = π*(y)/π_d(y)` to a common reference prior `π*`,
- **concept shift**   = the **residual** `I(Y; D | Z)` that survives *after* the label correction (differing label-generating mechanisms / diagnostic criteria across sites).

The glue is the **exact chain-rule identity (A1)** and the **tension theorem (A2)** showing the
encoder term and the decoder/concept term *fight* under label shift unless reweighted (A3/A4).
Target application: **psychiatric cross-site EEG DG**, where concept shift is not a corner case
but a first-order reality (diagnostic criteria, rater conventions, and inclusion thresholds genuinely
differ across clinical sites).

---

## 1. The three closest papers — what each does and what it lacks

### 1.1 Zhao et al., ICML 2019 ("On Learning Invariant Representations for DA", `zhao19a`)
**What it proves.** A *lower bound* (Thm 4.3) on the **joint** source+target error of any method
that learns a **marginally invariant** representation:
`ε_S(h∘g) + ε_T(h∘g) ≥ ½ (d_JS(D^Y_S, D^Y_T) − d_JS(D^Z_S, D^Z_T))²`.
The message: if marginal label distributions differ (`d_JS(D^Y_S,D^Y_T)>0`) and you drive the
marginal feature divergence `d_JS(D^Z_S,D^Z_T)→0`, the joint error is *forced up*. Marginal
invariance `I(Z;D)=0` is **harmful** under label shift. Also gives an upper bound (Thm 4.1) with a
conditional-shift term `min{E_S|f_S−f_T|, E_T|f_S−f_T|}`.

**What it lacks (for us).** It is a *negative* result about the *marginal* objective. It diagnoses the
disease; it does not prescribe a constructive multi-term objective. The "concept"/labeling-function
term appears only inside an upper bound as an *unestimable* `|f_S−f_T|` (depends on the unknown
optimal labeling function in feature space — they themselves flag this). 2-class, deterministic.

**Our relation.** Our encoder term is **conditional** `I(Z;D|Y)`, not marginal `I(Z;D)`. This is
*precisely* the quantity that side-steps Zhao19's lower bound: conditional invariance is compatible
with a perfect classifier even under label shift (a point GLS later formalizes). So Zhao19 is the
**motivation / impossibility wall** that tells us *not* to use the marginal term. We honor it, we do
not contradict it. **(This is the right framing — Zhao19 is a "do not do X" result, our pillar is "here
is the X-free decomposition.")**

### 1.2 Tachet des Combes et al., NeurIPS 2020 ("Conditional Distribution Matching and Generalized Label Shift", `GLS`)
**What it does.** Introduces **Generalized Label Shift (GLS)**: `D_S(Z|Y=y) = D_T(Z|Y=y) ∀y`,
i.e. label shift *in feature space* — exactly `I(Z;D|Y)=0` in our notation. Proves error
decomposition (Thm 3.1) and a joint-error bound (Thm 3.2: `ε_S+ε_T ≤ 2·BER`) under GLS. The
**operational engine**: estimate class-ratio importance weights `w_y = D_T(Y=y)/D_S(Y=y)` by a
moment-matching QP `w = C⁻¹μ` (confusion matrix `C`, target prediction marginal `μ`), then align the
**reweighted** source feature distribution `D_S^w(Z)` to `D_T(Z)`. This is the **anchor A4 (GLS
reweighting)** verbatim.

**What it lacks (for us).** Two structural gaps:
1. **No concept/decoder term.** GLS *assumes* conditional invariance is achievable and the residual
   `I(Y;D|Z)` is zero at the optimum (`ΔCE=0` under GLS). It has **no mechanism and no term** for the
   case where, *after* reweighting and conditional alignment, the label is still domain-dependent given
   Z. In clinical EEG that residual is real (different diagnostic criteria → `p(y|z)` differs by site).
2. **UDA, two domains, no source-free generalization.** GLS is unsupervised *domain adaptation*
   (one labeled source, one unlabeled target, target features available at train time). It is *not*
   domain *generalization* (their own "Extensions" explicitly lists DG as future work). Weight
   estimation `w=C⁻¹μ` needs target predictions.

**Our relation.** We **adopt GLS's reweighting as our label-shift correction (term 2)** and credit it
plainly. Our novelty sits *orthogonal* to GLS: (i) we add the **third, concept term** `I(Y;D|Z)` that
GLS structurally omits, and (ii) we operate in **multi-source DG** (no target at train) with a
label-conditional CMI estimator rather than per-pair adversarial feature alignment.

### 1.3 Zhao et al., NeurIPS 2020 ("Domain Generalization via Entropy Regularization", `entropyDG`)
**What it does.** DG (multi-source, unseen target). Argues marginal adversarial alignment gives only
`P(Z)` invariance, not `P(Y|Z)` invariance. Adds a **conditional-entropy regularizer** `H(Y|F(X))`
realized through K auxiliary per-domain classifiers in a minimax game. Thm 1: minimizing
`−H_{P_i}(Y|F(X))` ⇔ minimizing JSD of `{P_i(F(X)|Y=c)}_c` — i.e. it pushes toward conditional
invariance / matching `P(Y|Z)` across source domains. This is a **decoder-flavoured** regularizer:
it acts on the predictor's conditional, the same family as our concept term.

**What it lacks (for us).** **Theorem 1 explicitly assumes "all classes are equally likely"** (balanced
classes; they enforce it by "biased batch sampling" / weighted CE). That assumption is *exactly* the
label-shift case our framework is built to handle — and the case Zhao19 proves is dangerous. So
entropyDG's conditional-invariance guarantee **dissolves under label shift**, the regime that matters
clinically. It also has no covariate/label/concept *separation*: it is a single entropy knob, not a
decomposition, and gives no estimator of which shift is actually present.

**Our relation.** entropyDG is the closest *spiritual* relative of our **concept term** (decoder-side,
conditional). We differ by (i) **dropping the balanced-class assumption** via the GLS reweighting, (ii)
**naming the term as concept shift** `I(Y;D|Z)` inside an exact decomposition rather than a single
entropy penalty, and (iii) a **direct variational CMI estimator** instead of K adversarial auxiliary
classifiers (cheaper, K-free, and gives an actual nat-valued shift diagnostic).

---

## 2. The unifying object: where the three papers live in ONE identity

Anchor **A1 (exact, re-verified 1e-12 this pass):**

```
I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D)
```

This single chain-rule identity is the spine that contains all three works:

| Term | Our name | Who owns it | Their status |
|---|---|---|---|
| `I(Z;D)` (marginal) | (the trap) | Zhao19 lower bound | proven **harmful** under label shift → we never minimize it |
| `I(Z;D\|Y)` | **covariate** (encoder) | GLS (= `ΔCE=0` / GLS condition) | GLS minimizes it; entropyDG approximates it (balanced only) |
| `I(Y;D)` / `π_d` | **label** correction | GLS reweighting `w_y` (A4) | GLS owns the estimator; we adopt it |
| `I(Y;D\|Z)` | **concept** (decoder, residual) | — | **structurally absent in all three** |

**The tension theorem (A2/A3), re-verified this pass.** With a conditionally-invariant encoder
`I(Z;D|Y)=0`, A1 collapses to `I(Y;D|Z) = I(Y;D) − I(Z;D)`. Numerically reconfirmed on a constructed
domain-invariant-`p(z|y)` + per-domain-prior example:
`I(Y;D)=0.0823, I(Z;D)=0.0290, I(Z;D|Y)=0 (−2e-9), I(Y;D|Z)=0.0533 = I(Y;D)−I(Z;D)` and the identity
`I(Z;D|Y)−I(Y;D|Z) = I(Z;D)−I(Y;D)` holds to −0.053309444934 on both sides.
**Consequence:** under label shift `I(Y;D)>0`, zeroing the encoder term *forces a strictly positive
concept term* unless `I(Z;D)=I(Y;D)`, which (A3) requires zero Bayes error — impossible with
irreducible clinical label noise. **The encoder objective and the decoder objective fight.** A4 (GLS
reweighting) is the resolution: in the reweighted distribution `Ĩ(Y;D)=0`, decoupling the two CMIs so
both can be driven to zero.

This "they fight, and reweighting is what un-sticks them" statement is, to our reading, **not made
explicit in any of the three papers**. GLS gives the reweighting but never frames it as resolving a
*tension between two simultaneously-desired CMI terms* (it has only one). That framing is ours.

---

## 3. What is GENUINELY NOVEL vs INCREMENTAL (the honest ledger)

### Genuinely novel
1. **The three-shift CMI decomposition as one identity.** Covariate `I(Z;D|Y)`, label
   `I(Y;D)/π_d`, concept `I(Y;D|Z)` placed inside the exact chain-rule A1. None of the three papers
   carries all three terms; the concept residual is absent from all of them. The *unification* — that
   GLS, entropyDG, and Zhao19's trap are three faces of one identity — is a real conceptual contribution.
2. **The tension theorem (A2/A3) + its GLS resolution (A4) stated as a tension.** The explicit claim
   "conditional-invariance and concept-invariance are antagonistic under label shift, and label
   reweighting is exactly the operator that makes them compatible" is, as far as we can tell, new.
3. **Concept shift as a first-class, estimated quantity for psychiatric cross-site DG.** Treating
   `I(Y;D|Z)` as a *measurable diagnostic* of differing diagnostic criteria across sites — not an
   adversarial loss but a nat-valued readout — is application-novel and arguably the strongest selling
   point for a clinical-ML/AAAI audience.
4. **Dual variational estimators** for the encoder CMI and the decoder/concept CMI in the *DG*
   (source-free target) regime, with the label-conditional prior `π_y(D)` (our LPC-CMI), rather than
   per-pair adversarial alignment (GLS) or K auxiliary classifiers (entropyDG).

### Incremental / adopted (must credit, do not over-claim)
- **The encoder term `I(Z;D|Y)` itself = GLS in information-theoretic clothing.** GLS's
  `D_S(Z|Y)=D_T(Z|Y)` *is* `I(Z;D|Y)=0`. We must say so plainly. Our contribution here is the
  estimator and the DG setting, **not** the concept of conditional feature invariance.
- **The label-shift reweighting (A4) is GLS's, essentially Combes 2020 / Lipton 2018.** We adopt
  `w_y=π*(y)/π_d(y)` wholesale. No novelty claimed on the weight estimator.
- **A decoder/conditional regularizer in DG is foreshadowed by entropyDG** (and CIDDG/Li 2018). Our
  concept term is in the same family; novelty is the label-shift-robust version + the decomposition
  membership, not the bare idea of regularizing `P(Y|Z)`.
- **The marginal-invariance-is-harmful insight is Zhao19's**, not ours. We inherit it as motivation.

### Risk: the "just MI bookkeeping" objection
A skeptical reviewer can say A1 is "a one-line chain-rule of mutual information." **True** — the
identity is elementary. The defensible contribution is *not* the algebra but (a) mapping each term to a
named, separately-estimated shift, (b) the tension theorem turning the algebra into a
*training-dynamics* statement, and (c) demonstrating the concept term is non-zero and *matters* on real
psychiatric cross-site EEG. **If the empirical concept-shift story is weak, the paper collapses to "we
re-derived GLS and added an entropy term." Pillar 4's viability is hostage to the experiments.**

---

## 4. AAAI / ICLR viability — honest assessment

**Verdict: viable for AAAI-27 Main Technical Track as positioned, conditional on the empirical concept-shift evidence. Borderline for ICLR without a tighter theorem or a state-of-the-art benchmark win.**

Reasoning:

- **For AAAI (application-forward, methodological-soundness venue):** The combination of (i) a clean
  unifying identity, (ii) the tension theorem, and (iii) a *clinically real* concept-shift application
  is a good fit. AAAI rewards a well-motivated, correctly-positioned framework with a believable use
  case more than a raw SOTA delta. The psychiatric cross-site framing is a genuine differentiator —
  GLS/entropyDG used MNIST/PACS/Office, none addressed concept shift from diagnostic-criteria drift.
  **This is our lane.** The verified leakage estimator (rock-solid per empirical-findings memory) and
  the GLS-parity / SCPS 4/4 results give the methods section a spine.

- **For ICLR (theory/novelty bar higher):** The honest risk is that each *individual* ingredient is
  prior art (GLS term, GLS reweighting, entropy-style decoder term, chain-rule identity). ICLR
  reviewers are quick to label a recombination "incremental." To clear ICLR we would need *at least one*
  of: (a) a **new generalization bound** in terms of the three CMIs (analogous to Thm 3.1 but with the
  concept residual carried explicitly, and *tighter* or *more estimable* than Zhao19's `|f_S−f_T|`),
  or (b) a **clear benchmark win** over IWDAN/IWCDAN/entropyDG on a shared protocol with non-trivial
  label *and* concept shift. The exact identity alone will not clear ICLR novelty.

**Concrete must-do's to lock the positioning (in priority order):**
1. **Show the concept term is non-zero and consequential** on the real psychiatric cross-site data —
   measure `I(Y;D|Z)` after GLS reweighting + encoder-invariance, demonstrate residual > 0 and that
   penalizing it improves target accuracy/calibration. *This is the linchpin.*
2. **State the tension theorem as a formal proposition** (A2/A3) in the paper, with the constructed
   example as a figure — it is the one piece reviewers cannot attribute to prior work.
3. **Credit GLS and entropyDG explicitly and early** (a positioning paragraph + the Table in §2 of
   this note). Pre-empting "isn't this just GLS?" in the intro is worth more than hiding it.
4. **Frame Zhao19 as the wall we route around**, not a competitor — `I(Z;D|Y)` vs `I(Z;D)`.
5. If aiming ICLR: derive the three-CMI generalization bound (item (a) above). If AAAI-only: the
   decomposition + tension + clinical demonstration suffices.

**Bottom line.** The framework is *correctly* positioned and *honestly* novel in its **unification +
tension + concept-term-for-psychiatry** axis, while **transparently incremental** on each isolated
component. That is a defensible AAAI paper. It is not yet an ICLR paper until either a new bound or a
benchmark win is in hand. Do not oversell the chain-rule identity as the contribution; sell the
decomposition's *use* (separable, estimable, clinically meaningful shift diagnostics) and the tension
theorem.
