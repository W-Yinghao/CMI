# OACI — theory

Three results plus two estimator/estimand corrections, each stated precisely with its
**proof obligation** and **what it does NOT claim**. The discipline is the closed line's
hard-won lesson: never assert a theorem stronger than the assumptions buy (cf.
`h2cmi/THEORY.md` P0-2/P0-3). This file supersedes the v1 scaffold, which over-claimed in
four places — those corrections are folded in below and marked **[C1]–[C4]**.

Notation (matches [`README.md`](README.md)): eligibility ``S_y = {d : n_{d,y} ≥ m}``,
overlap-aware leakage estimand reported under a **fixed** reference prior ``p_ref``.

---

## §0 — three support notions (do not conflate) **[C2]**

* **Structural support** — the DGP can produce class ``y`` in domain ``d`` at all
  (``p(z|y,d)`` is *defined*). Unobservable; never asserted by the code.
* **Observed support** — ``n_{d,y} > 0`` (`present`). A sample was seen.
* **Estimator-eligibility** — ``n_{d,y} ≥ m`` (`eligible`). Enough samples to estimate the
  conditional at the tolerance the critic/test needs. This is a **finite-sample variance
  guard**, NOT a population identifiability fact: a cell can be structurally supported yet
  ineligible (too few samples), and ``m`` trades coverage against estimator variance.

The support graph thresholds on **eligibility**. The paper must state results at the
population level (structural/observed support) separately from the finite-sample eligibility
that the algorithm actually keys on; an eligibility-driven "non-identifiable" flag is a
*we-cannot-estimate-here* statement, not *the-quantity-does-not-exist*.

---

## §1 — Support theorem (partial identifiability) **[C1]**

**Setup.** Domains ``d ∈ {1..K}``, classes ``y ∈ {1..C}``, eligibility sets ``S_y``.

**Statement.**
1. **Per-class identifiability.** The conditional-invariance equality
   ``p(z|y,d) = p(z|y,d')`` for a **fixed** class ``y`` is **population-identifiable** iff both
   cells have **positive structural support** for class ``y`` (``p(y,d)>0`` and ``p(y,d')>0``);
   it is **estimable at finite sample** iff both cells clear the operational ``m``-gate
   (``d, d' ∈ S_y``). Identifiability is the population property; eligibility (``S_y``) is the
   algorithm's finite-sample proxy for it. (Module: `is_estimable_pair(d, d', y)` — the
   ``m``-gate; deprecated alias `is_identifiable_pair`.)
2. **No cross-class transitive reach.** A path of *other* classes does **not** transfer a
   fixed-``y`` equality. If neither ``d`` nor ``d'`` has structural support for ``y`` jointly,
   then ``p(z|y,d)=p(z|y,d')`` is not identifiable even when ``d`` and ``d'`` are linked through
   shared *other* classes. The set of identifiable equalities is exactly
   ``⋃_y { (d,d') : p(y,d),p(y,d')>0 }`` — **not** its transitive closure across classes; the
   estimable subset replaces ``>0`` with the ``m``-gate.
3. **Coupling ≠ identifiability.** The domain–domain graph (``d ~ d'`` iff they co-observe
   some eligible class) and its connected components describe only **decomposability**: a
   single shared encoder makes the per-class constraints in one component share parameters,
   so the optimization does not separate across a component; across components it does. This
   is an optimization-structure statement and carries **no** identifiability content for any
   particular equality. (Module: `coupled(d, d')`, `coupling_components`.)

**Proof obligation.** (1): with no structural support for ``(d,y)`` the conditional
``p(z|y,d)`` is undefined, so the equality is not even stated; with positive support for both,
a grouped two-sample divergence is consistent as ``min(n_{d,y}, n_{d',y}) → ∞`` — and the
``m``-gate is just the finite-sample decision of *when we have enough samples to run that test*.
(2): exhibit two laws agreeing on every structurally-supported cell (equal likelihood) but
disagreeing on a fixed-``y`` cross-pair where one domain has zero support for ``y`` — no test
separates them, so the equality is non-identifiable regardless of connectivity. (3): the encoder ties parameters along graph edges; show the
KKT system block-diagonalises across components (independent duals), while within a
component the identifiable-constraint set is still only the per-class pairs of (1)–(2).
**This is the core paper theorem; [`support_graph.py`](support_graph.py) is its constructive
content.**

**Does NOT claim.** That enforcing the identifiable equalities helps target accuracy (that
is empirical, `EXPERIMENTS.md`). That connectivity licenses imputing or aligning a
cross-component / ineligible cell.

---

## §2 — No-excess empirical risk (noninferiority of the feasible solution)

**Setup.** Stage 1: ERM gives ``R_ERM`` (achieved source risk on a held-out source split).
Stage 2: ``min_θ UCB_{1-α}[L_Q^ov]`` s.t. ``R_src(θ) ≤ R_ERM + ε``.

**Statement.** Any feasible ``θ`` has ``R_src(θ) ≤ R_ERM + ε``; the ERM solution is feasible
(for ``ε ≥ 0``), so stage 2 returns a solution that is ε-noninferior to ERM on source risk
with leakage no larger than ERM's.

**Proof obligation.** Immediate given exact feasibility — the real content is the **caveat**:
the dual/penalty solver enforces the constraint only up to optimization slack ``δ_opt``, and
``R_ERM`` is itself estimated (generalization gap). The honest claim is ε-noninferiority
**modulo** ``δ_opt`` and the ``R_ERM`` estimation error; we **report the realized source-risk
gap**, never assume it equals ``ε``.

**Does NOT claim.** Anything about *target* risk. ε-noninferiority on source ≠ target
improvement. An ERM-optimality constraint + DG penalty is **not** itself the novelty (it
exists in prior work) — §2 is the safety floor under §1, not the contribution.

---

## §3 — Label-preservation proposition (no *explicit* Y-erasure) **[C3]**

**Setup.** "Aligning" an ineligible/cross-component cell means coercing ``p(z|y,d)`` toward a
reference (uniform prior, marginal ``p(z|d)``, chain target) where ``d ∉ S_y``. The LPC
synthetic study showed these coercions erase label info (marginal → label-erasure, chain →
Y-erasure, uniform → mis-specification; `notes/synthetic_results.md`).

**Statement (corrected, weaker).** OACI adds **no explicit alignment term** that touches an
ineligible or singleton-support cell: such cells contribute zero terms to the objective, so
the optimizer never *explicitly forces* ``p(z|y,d)`` toward a ``Y``-independent target there.
That is the defensible claim — OACI avoids the *forced* Y-erasure of the uniform/marginal/
chain routes.

**What is NOT claimed [C3].** That the representation on that cell is "left free" or that its
label separability is lower-bounded by ERM. Under a **shared encoder**, gradients from
*other* (eligible) cells still move the representation at the untouched cell, so its
``∂(penalty)/∂(parameters)`` is **not** identically zero in general. The strong "free / ERM
lower-bounded" form holds only under additional assumptions — e.g. a **cell-separable
parametrization** (per-cell heads), or a locality/capacity condition bounding cross-cell
gradient leakage. The paper states the weak form as the result and the strong form as a
corollary-under-assumption, with the assumption named.

**Proof obligation.** Weak form: show the penalty has no term indexed by an ineligible/
singleton cell (direct from the objective's construction over `overlap_terms()`), hence no
*explicit* gradient toward a Y-collapsing target on it. Strong form: under cell-separable
parameters the only force on that cell's head is the task loss ⇒ label separability ≥ ERM;
exhibit the uniform/marginal/chain penalties' nonzero coercive gradient on the same cell as
the contrast.

---

## §4 — Estimator correction: the probe gap is a LOWER bound on CMI **[C4]**

For any probe ``q(D|Z,Y)``,
```
E[-log q(D|Z,Y)] = H(D|Z,Y) + E_{Z,Y} KL( p(·|Z,Y) ‖ q(·|Z,Y) )  ≥  H(D|Z,Y),
```
so the probe entropy-gap ``Ĥ(D|Y) − E[-log q]`` **under**estimates ``I_ov`` — it is a *lower*
bound, with slack equal to the mean posterior KL. Therefore:

* We do **not** claim, and never report, a UCB on the true ``I_ov`` (the precise LPC error:
  a posterior plug-in relabelled an upper bound).
* The functional we minimise and bound is the **probe-class-extractable leakage**
  ```
  L_Q^ov = sup_{q∈Q} ( Ĥ(D|Y) − E[-log q(D|Z,Y)] )      over comparable cells,
  ```
  the leakage a fixed probe family ``Q`` can extract ("extractable conditional domain
  information", the project's binding name). It is a sound *lower* bound on ``I_ov`` and an
  honest operational target (you cannot exploit what no admissible probe extracts).
* ``UCB_{1-α}[L_Q^ov]`` is an upper confidence bound on **that** functional, via a bootstrap
  clustered on the dependence unit (recording/subject). The capacity ``sup`` over ``Q`` is
  **part of the estimator**, so it is re-selected **inside each resample**
  (`capacity_selection_in_resample`); selecting capacity once on the full sample and then
  bootstrapping is optimistic. Cross-fit + within-``(Y, d-group)`` permutation null guard the
  point estimate; negatives are kept (no truncation bias).

Config carries this naming: `UCBConfig.functional = "extractable_LQ_ov"`.

---

## §Estimand — fixed reference prior, L_abs vs L_cond **[C4-weighting]**

`overlap_terms()` weights each comparable class by a **fixed** ``p_ref(y)`` (passed across the
whole missing-cell sweep), not a per-configuration renormalisation, so the estimand does not
drift as cells are deleted. Two estimands:
```
L_abs  = Σ_{y∈C_cmp} p_ref(y)           · L_y      (PRIMARY)
L_cond = Σ_{y∈C_cmp} p_ref(y | y∈C_cmp) · L_y      (diagnostic)
```
`L_cond` renormalises onto the comparable classes, so its weights **move** as support
fragments (a different question at each deletion level) — useful as a diagnostic, never the
headline. The main result reports **``L_abs`` together with the identifiable mass fraction**
``Σ_{y∈C_cmp} p_ref(y)`` (`identifiable_mass_fraction()`), so a leakage number is always
paired with how much reference mass it actually covers. When support fragments, ``L_abs``
falls because comparable mass leaves the sum — and that drop is reported as the honest
consequence of non-identifiability, not hidden by renormalisation.

---

## Guard-rails carried over from the closed line (binding)

* The leakage number is **"extractable conditional domain information"** / ``L_Q^ov``
  (grouped, capacity-sup, cross-fit, perm-null), never "precise CMI / ``I(Z;D|Y)``".
* ``UCB_{1-α}[L_Q^ov]`` must be an actual confidence bound (clustered bootstrap, capacity
  selection inside the resample). A posterior-KL is not a UCB; a probe gap is a lower bound.
* Any "invariance" claim is checked against the **collapse** alternative (does leakage fall
  because capacity was lost, not because the representation became domain-balanced?).
* Identifiability is **per class**; connectivity is only decomposability.
