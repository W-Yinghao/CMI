# TOS-CMI — theory

Task-Orthogonal Selective CMI. The one-line idea: do not erase **all** conditional
domain information `I(Z;D|Y)`; erase it only on the subspace that is **domain-rich and
label-light**, and **refuse** (identity) when no such subspace is risk-feasible.

This is a direct algorithmic response to three established negative results in this repo
(see `archive/lpc-cmi-failed/` and `notes/EVIDENCE_LEDGER.md`):

* **global LPC collapses TSMNet** — driving `I(Z;D|Y)→0` over the whole representation
  destroys the SPD geometry;
* **λ-sensitivity on 2a** — large penalty weight removes label-bearing structure;
* **uneven leakage** — in GraphCMI the leakage is concentrated in some channels/nodes,
  not spread uniformly, so a single global knob is the wrong instrument.

---

## 1. The two Fishers

Work at a fixed representation `Z ∈ R^d` (a layer/channel block). Define two
between-group scatter matrices — how far group means sit from the (conditional) grand
mean, weighted by group probability:

```
F_Y     = Σ_y p(y)  (μ_y − μ)(μ_y − μ)^T                                  [d×d]
F_{D|Y} = Σ_y p(y)  Σ_d p(d|y) (μ_{d,y} − μ_y)(μ_{d,y} − μ_y)^T           [d×d]
```

`F_Y` is the usual class between-scatter (label signal). `F_{D|Y}` is the **class-
conditional** between-domain scatter: it only sees domain spread *after conditioning on
the label*, which is exactly the quantity `I(Z;D|Y)` penalises. A unit direction `v`
with large `v^T F_{D|Y} v` carries conditional domain information; with large
`v^T F_Y v` it carries label information. Code: [`fisher.py`](fisher.py).

## 2. The generalized eigenproblem

```
F_{D|Y} v_j = ρ_j (F_Y + η I) v_j
```

`B = F_Y + ηI` is SPD, so this is a stable symmetric generalized problem
(`scipy.linalg.eigh(A,B)`). Large `ρ_j` ⇒ direction `v_j` is **domain-rich relative to
its label content** — domain-rich, label-light. Code: [`subspace.py`](subspace.py)
`solve_generalized`.

> **Why `ρ` alone is not enough (the trap that the failed line would have walked into).**
> In a *label-null* direction `B ≈ ηI`, so `ρ = dom/(η)` blows up for any sampling-noise
> domain energy. Ranking by `ρ` then selects pure-noise directions. This is precisely the
> "subspace ranking unstable across seed/probe/fold" failure mode named in the project's
> termination condition. We defuse it in §4.

## 3. The selective penalty

Let `P_N` be the Euclidean orthogonal projector onto the span of the **selected**
directions (§4). Train with

```
L = CE(Z) + λ · I( P_N Z ; D | Y )
```

The classifier still sees the full `Z`; only the *invariance pressure* is restricted to
`P_N Z`. The task-entangled complement `(I − P_N) Z` is never pushed toward invariance,
so label-bearing domain structure there is left intact. The leakage term is the same
label-prior-corrected posterior-KL plug-in used by the AAAI core
(`E_i KL(q_ψ(D|z_{N,i}, y_i) ‖ π_{y_i}(D))`, tight at the Step-A critic optimum). Code:
[`selective_cmi.py`](selective_cmi.py).

## 4. Risk-feasibility, the null floor, and the identity fallback

A direction `j` (unit eigenvector `u_j`, energies `dom_j = u_j^T F_{D|Y} u_j`,
`lab_j = u_j^T F_Y u_j`) is **nuisance-eligible** iff *all three* hold:

1. **domain-rich**   `dom_j / (lab_j + η) ≥ τ_ρ`;
2. **label-light**   `lab_j ≤ ε_label · max_k lab_k`  — the **risk-feasibility gate**:
   a direction is only deletable if removing it costs almost no label information;
3. **above noise**   `dom_j ≥ max( dom_floor·max_k dom_k , safety · floor_null )`,
   where `floor_null` is the largest `F_{D|Y}` eigenvalue under a **within-`Y`
   permutation of `D`** (the sampling-noise floor; same permutation-null philosophy as
   `cmi/eval/leakage_audit.py`). This is what kills the label-null/noise directions from
   §2.

`P_N` is the projector onto the eligible span (capped at `max_dim`). **If no direction
qualifies, `P_N = 0`: the penalty is identically zero and the method degrades to
identity.** Refusing to delete is a first-class, falsifiable behaviour — it is what
global LPC structurally cannot do. Code: [`subspace.py`](subspace.py) `select_nuisance`,
`SubspaceSelector`.

## 5. Proposition (Bayes-risk preservation)

**Setup.** Let `Z = (Z_Y, Z_N)` (an orthogonal split). Assume

* **(A) label sufficiency of `Z_Y`**:  `I(Y; Z_N | Z_Y) = 0`  (equivalently
  `Y ⊥ Z_N | Z_Y`, i.e. `p(Y | Z_Y, Z_N) = p(Y | Z_Y)`);
* **(B) leakage confinement**:  `I(Z_Y; D | Y) = 0`  (all conditional domain leakage
  lives in `Z_N`).

**Claim.** Projecting out `Z_N` (using `Z_Y` only) leaves the Bayes risk unchanged and
removes the conditional domain leakage:

```
R*(Z_Y) = R*(Z_Y, Z_N)        and        I(Z_Y; D | Y) = 0.
```

**Proof.** The Bayes risk of a representation `R*` is a functional of the posterior
`p(Y | ·)`. By (A), `p(Y | Z_Y, Z_N) = p(Y | Z_Y)`, so the Bayes-optimal decision rule
and its risk are identical whether or not `Z_N` is observed: `R*(Z_Y) = R*(Z_Y, Z_N)`.
Leakage-freeness of the kept part is (B). ∎

**Converse / why selection (not global erasure) is necessary.** If (A) fails — task and
domain subspaces *overlap*, so `Z_N` carries label information given `Z_Y` — then
`H(Y | Z_Y) > H(Y | Z_Y, Z_N)` and removing `Z_N` **raises** the Bayes risk. A global
penalty that drives `I(Z;D|Y)→0` pays this cost unconditionally (the TSMNet collapse).
The risk-feasibility gate (§4.2) detects exactly this overlap — the domain-rich
directions are then also label-rich, fail `label-light`, and the method returns identity
instead of paying the risk.

**Empirical instrument.** [`eval/proposition.py`](eval/proposition.py) `bayes_risk_check`
reports, on a held-out split, label accuracy on `Z` vs `(I−P_N)Z` (should be equal at
overlap 0) and conditional-domain leakage on `P_N Z` vs `(I−P_N)Z` (high vs ≈0). Under
overlap the selector returns identity, so accuracy is preserved by construction. The
synthetic world [`data/synthetic.py`](data/synthetic.py) realises (A)+(B) at
`overlap=0` and the worst case (B-confinement broken, domain collinear with the class
discriminant) in `make_collinear`.

## 6. The termination gate (when to abandon this direction)

The selected subspace must be **the same object** across seeds, probes and folds — else
we are fitting noise. [`eval/stability.py`](eval/stability.py) measures pairwise
principal-angle overlap between selected `P_N` bases. The stated termination conditions:

* selection unstable (`mean pairwise subspace overlap` low, or `n_identity` mixed across
  seeds) — **stop**, this is just a more complicated regularizer;
* cannot beat global LPC on the clearest collapse case (TSMNet) — **stop**.

The synthetic suite already exercises the gate: a clear signal gives `overlap > 0.9`
across 5 seeds; a pure-noise world gives identity on every seed (no false subspace).

## 7. What this is and is not

A **research implementation on a simulator**: every piece is correct, differentiable,
null-calibrated, composes end-to-end, and the proposition is what the code computes. It
is **not** a real-EEG result. The confirmatory protocol lives in
[`INTEGRATION.md`](INTEGRATION.md) (TSMNet/2a/GraphCMI, the hardest counterexamples,
LOSO × seeds, source-only λ selection).
