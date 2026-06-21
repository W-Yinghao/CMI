# OACI — theory

Three results, each stated precisely with its **proof obligation** and, crucially, **what
it does NOT claim**. The discipline here is the LPC line's hard-won lesson: do not assert a
theorem stronger than the assumptions buy (cf. `h2cmi/THEORY.md` P0-2/P0-3, where a
posterior-KL was wrongly called an upper bound and a "CMI=0 ⇔ zero Bayes error" theorem was
false). Notation matches [`README.md`](README.md): support set `S_y = {d : n_{d,y} ≥ m}`,
overlap-aware leakage `I_ov(Z;D|Y) = Σ_y p(y) I(Z; D_{S_y} | Y=y, D∈S_y)`.

---

## §1 — Support theorem (partial identifiability)

**Setup.** Domains `d ∈ {1..K}`, classes `y ∈ {1..C}`, i.i.d. cells with effective counts
`n_{d,y}`. The *support graph* `G` has the `K` domains as nodes; an edge `d ~ d'` iff
`∃ y : d, d' ∈ S_y` (they co-observe some class with estimable support). Components of `G`
are computed in [`support_graph.py`](support_graph.py).

**Statement.**
1. The conditional-invariance constraint `p(z|y,d) = p(z|y,d')` is identifiable from data
   **iff** `(d,y)` and `(d',y)` are both observed cells (`d, d' ∈ S_y`) — i.e. only on
   *comparable* cells.
2. The joint statement "a single representation makes `p(z|y,·)` invariant across a set `U`
   of domains" is identifiable **iff** `U` lies within one connected component of `G`.
   Across components there is **no** observed class linking the groups, so cross-component
   invariance is **not** determined by the data and requires an extra **untestable**
   assumption (a parametric / extrapolation tie across the support gap).

**Proof obligation.** (1) is immediate: with no joint sample of `(d,y)` the conditional
`p(z|y,d)` has no consistent estimator, so the equality is neither testable nor enforceable;
with both observed, a two-sample test/divergence is consistent as `min(n_{d,y}, n_{d',y})→∞`.
For (2), build the constraint hypergraph: each comparable class `y` ties the domains in
`S_y`; transitive closure of these ties is exactly graph connectivity, and any tie crossing
two components is unsupported by every class. Formalize "untestable" as: there exist two
data-generating laws agreeing on all observed cells (hence equal likelihood) yet disagreeing
on the cross-component conditional — so no test separates them. **This is the core paper
theorem; the [`support_graph`](support_graph.py) module is its constructive content.**

**Does NOT claim.** It says nothing about whether enforcing the *identifiable* invariances
helps target accuracy (that is empirical, §EXPERIMENTS). It does not license imputing the
non-identifiable cells.

---

## §2 — No-excess empirical risk (noninferiority of the feasible solution)

**Setup.** Stage 1 fits ERM to get `R_ERM = min_θ R_src(θ)` (under the chosen risk metric;
in practice the achieved ERM value on a held-out source split). Stage 2 solves
`min_θ UCB_{1-α}[I_ov(Z;D|Y)]` subject to `R_src(θ) ≤ R_ERM + ε`.

**Statement.** Any feasible `θ` satisfies `R_src(θ) ≤ R_ERM + ε`; in particular the stage-2
optimum increases source risk by at most `ε` relative to ERM. If the feasible set is
nonempty (it contains the ERM solution itself when `ε ≥ 0`), stage 2 returns a solution that
is **ε-noninferior** to ERM on source risk while having no larger leakage UCB than ERM
(since ERM is feasible, the leakage-minimizer can only match or beat it).

**Proof obligation.** Trivial given exact constraint satisfaction — the content is the
**caveat that practice does not satisfy it exactly**: the dual/penalty solver only
approximately enforces the constraint, and `R_ERM` is itself estimated. The honest version
states the bound *modulo* (a) optimization slack `δ_opt` on the constraint and (b)
generalization gap of `R_ERM`. We must report the realized source-risk gap, not assume `ε`.

**Does NOT claim.** Nothing about *target* risk. ε-noninferiority on source ≠ improvement on
target; the target story is the worst-domain / calibration evidence in `EXPERIMENTS.md`.
Prior work already pairs an ERM-optimality constraint with a DG penalty — §2 alone is **not**
the novelty (see README "why this is new"); it is the safety floor under §1.

---

## §3 — Label-preservation proposition (no forced Y-erasure)

**Setup.** "Aligning" an unsupported cell means coercing `p(z|y,d)` toward some reference
(uniform prior, marginal `p(z|d)`, or a chain target) where `S_y` does not actually contain
`d`. The LPC synthetic study showed these routes erase label information: marginal alignment
→ label-erasure, chain → Y-erasure, uniform → mis-specification (`notes/synthetic_results.md`).

**Statement.** Restricting alignment to comparable cells (`d ∈ S_y`) leaves the per-class
conditionals on unsupported cells **free**, so the objective imposes no constraint that
forces `Z ⊥ Y` on those cells. Concretely: if a class `y` is singleton-support (`|S_y|=1`)
or `(d,y)` is unobserved, OACI adds **no** term touching `p(z|y,d)`, hence cannot push it
toward a `Y`-independent target. Therefore the optimum's label separability is lower-bounded
by ERM's on exactly the cells the global/uniform routes would have erased.

**Proof obligation.** Show the OACI penalty gradient w.r.t. parameters governing an
unsupported cell is identically zero (the cell contributes no term), so the only force on it
is the task loss — which preserves `Y`. Contrast: write the uniform/marginal/chain penalties
and exhibit their nonzero gradient pushing toward a `Y`-collapsing target on the same cell
(the synthetic counterexample, lifted to a proposition).

**Does NOT claim.** It does not say leaving cells free is *optimal* for target accuracy —
only that it avoids a known harmful coercion. It does not prevent collapse arising from the
task loss or capacity; that is monitored empirically (eff-rank / utility gates, as in the
LPC P1.5 audit).

---

## Guard-rails carried over from the closed line (binding)

* The leakage number reported is **"extractable conditional domain information"** (grouped,
  max-probe, cross-fit, perm-null), never "precise CMI / `I(Z;D|Y)`".
* `UCB_{1-α}[I_ov]` must be an actual confidence bound (clustered bootstrap + capacity
  sweep). A posterior-KL is **not** a UCB.
* Any "invariance" claim must be checked against the **collapse** alternative (does leakage
  fall because the representation lost capacity, not because it became domain-balanced?).
