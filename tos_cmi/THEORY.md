# TOS-CMI — theory

Task-Orthogonal Selective CMI. The one-line idea: do not erase **all** conditional
domain information `I(Z;D|Y)`; erase it only on the subspace that is **domain-rich and
label-light**, and **refuse** (identity) when no such subspace is risk-feasible.

> **Scope / naming caveat (read first).** The *current implementation* selects the subspace
> from **first-moment (between-group mean) scatter** statistics. So what it actually
> certifies is a **label-mean-scatter-light** subspace, not a fully **task-orthogonal** one:
> it is blind to task/domain information carried in covariance/SPD geometry, higher moments,
> or nonlinear interactions ([`tests/test_limits.py`](tests/test_limits.py) is the explicit
> covariance-only counterexample where it correctly no-ops). The aspirational
> "task-orthogonal" name requires the **score-Fisher / gradient-conflict** version (§8,
> deferred). Read every "Fisher" below as a *first-moment linear proxy*, every `domadv` as a
> *linear-probe advantage*, and the whole package as a **synthetic proof-of-concept**, not
> EEG evidence.

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

`F_Y` is the class between-MEAN scatter (a first-moment label-signal proxy). `F_{D|Y}` is
the **class-conditional between-domain MEAN scatter** — a **first-moment linear proxy for**
`I(Z;D|Y)`, **not** the CMI itself (it equals neither `I(Z;D|Y)` nor a bound on it; it
vanishes whenever domains differ only in covariance/higher moments, even if `I(Z;D|Y)>0`).
A unit direction `v` with large `v^T F_{D|Y} v` carries *mean-shift* conditional domain
information; with large `v^T F_Y v` it carries *mean-shift* label information. Both miss
covariance/nonlinear/synergistic structure (the score-Fisher version in §8 targets that).
Code: [`fisher.py`](fisher.py).

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

> **Algorithm–theorem gap (honest).** §5 reasons about *removing* `Z_N`, but training keeps
> the full `Z` and only penalises `I(Z_N;D|Y)`. These differ: the exact chain rule is
> `I(Z;D|Y) = I(Z_T;D|Y) + I(Z_N;D|Y,Z_T)`, so even with `I(Z_T;D|Y)=I(Z_N;D|Y)=0` the joint
> `Z` can still leak `D` through *synergy*. The theoretically-aligned objective is therefore
> `I(P_N Z; D | Y, sg(P_T Z))` — the critic conditioned on the (stop-gradient) task component
> as context — which needs a full critic plus a task-only baseline critic, not a plain KL to
> `p(D|Y)`. The current code implements the simpler `I(P_N Z;D|Y)`; the conditional-on-task
> form is part of the §8 redesign.

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

**Loss-dependence of the converse (corrected).** The forward claim above holds for any
proper loss. The *converse* — "if (A) fails, removing `Z_N` strictly raises risk" — is
loss-dependent:

* **log-loss** is exact: `R*_log(Z_Y) − R*_log(Z_Y,Z_N) = I(Y; Z_N | Z_Y) ≥ 0`, with strict
  inequality iff (A) fails.
* **0–1 risk** gives only `R*_01(Z_Y) ≥ R*_01(Z_Y,Z_N)`, **not** necessarily strict: extra
  information can change the posterior without changing the `argmax`, so the Bayes decision —
  and its 0–1 risk — may be unchanged even when `I(Y;Z_N|Z_Y)>0`.

Either way the *direction* is right: removing a label-bearing `Z_N` never *helps* and can
hurt. A global penalty that drives `I(Z;D|Y)→0` pays this cost unconditionally (the TSMNet
collapse); the risk-feasibility gate (§4.2) detects the overlap (domain-rich directions are
then also label-rich, fail `label-light`) and returns identity.

**Empirical instrument.** [`eval/projection_ablation.py`](eval/projection_ablation.py)
`linear_probe_projection_ablation` estimates `P_N` on a **selector-train** split and reports,
on a **disjoint probe-test** split (this is the fix for the earlier selection-leakage bug —
test labels/domains no longer enter `P_N`), linear label accuracy on `Z` vs `(I−P_N)Z` and a
linear conditional-domain *advantage* on `P_N Z` vs `(I−P_N)Z`. These are linear-probe
diagnostics, **not** CMI. The synthetic world [`data/synthetic.py`](data/synthetic.py)
realises (A)+(B) at `overlap=0`, the worst case (domain collinear with the class
discriminant) in `make_collinear`, and the **first-moment blind spot** (covariance-only
domain leakage) in `make_covariance_only`.

## 6. The termination gate (when to abandon this direction)

The selected subspace must be **the same object** across seeds, probes and folds — else
we are fitting noise. [`eval/stability.py`](eval/stability.py) gates on the **dimension-
sensitive projection distance** `‖P_1 − P_2‖_F` (a true metric; 0 iff identical projectors),
plus dimension spread and identity-decision consistency. It also reports a containment-biased
`cos²`-similarity for context only — that one is *not* a metric and reads 1 when a smaller
span sits inside a larger one (so `0.99` cos²-similarity with `k∈{2,3}` is **not** "stable").
Recovery vs a known span is reported as **precision/recall** (`precision_recall`), since
selecting 2 of a planted 4-D span is precision≈1 but recall≈0.5 — not "recovered the
subspace". The stated termination conditions:

* selection unstable (large `proj_dist_max`, `k_spread` wide, or `n_identity` mixed across
  seeds) — **stop**, this is just a more complicated regularizer;
* cannot beat global LPC on the clearest collapse case (TSMNet) — **stop**.

## 7. What this is and is not

A **synthetic-only research prototype**: every piece is correct, differentiable,
null-calibrated, composes end-to-end, and the proposition (§5) is what the code computes —
on a simulator whose structure matches the method's first-moment assumptions. It is **not**
a real-EEG result, and **not** yet wired into the trainer/TSMNet/2a/GraphCMI (that is a
plan in [`INTEGRATION.md`](INTEGRATION.md), not a result). The confirmatory protocol there
requires LOSO × seeds, source-only λ selection, and beating global LPC on the collapse case.

## 8. Known limitations and the path to actual novelty

The reviewer's points, recorded so they are respected rather than glossed:

1. **First-moment only.** `F_Y`, `F_{D|Y}` are mean scatters; they miss covariance/SPD,
   higher-moment, and nonlinear task/domain information ([`tests/test_limits.py`](tests/test_limits.py)).
   ⇒ "label-mean-scatter-light", not "task-orthogonal".
2. **The "risk-feasible" gate is not yet a risk bound.** `lab_j ≤ ε·max lab` is a relative
   label-scatter threshold: it is per-direction (not span-level), coordinate-sensitive, and
   constrains no CE/accuracy/source-risk quantity. A real gate is a source-only validation
   bound, e.g. `UCB_{1−α}[ R_val((I−P_N)Z) − R_val(Z) ] ≤ δ`, checked on data **not** used
   for the Fisher estimation.
3. **Gradient conflict is not implemented.** The original vision ("task/domain gradient
   conflict decides the deletable subspace") needs *score* Fishers
   `G_Y=E[g_Y g_Y^T]`, `G_{D|Y}=E_Y E[g_D g_D^T|Y]` from `g_Y=∇_z log p_θ(Y|z)`,
   `g_D=∇_z log q_ψ(D|z,Y)`, solved as `G_{D|Y} v = ρ (G_Y + η M) v` with a whitening metric
   `M` (representation covariance) instead of bare `I`; plus a **parameter-level** conflict
   step (PCGrad-style projection of the CMI gradient off the task gradient), since a
   representation-space `P_N` does not guarantee the shared encoder update spares the task.
4. **Conditional-on-task objective.** `I(P_N Z; D | Y, sg(P_T Z))` (§3 gap), not `I(P_N Z;D|Y)`.
5. **Synthetic is too aligned.** Needs class-specific domain shift, imbalanced/missing
   `(d,y)` cells, covariance-only and nonlinear/XOR leakage, classifier-preserving
   rescalings, a rotating carrier during training, and `rank(nuisance) ⋛ k`.

**Prior-art delta (why this is not just SCA/ISR/LEACE).** Class-conditional scatter +
generalized eigenproblem is Scatter Component Analysis; class-conditional first moments for
invariant/spurious subspaces is ISR; minimal-damage linear concept erasure is LEACE, and
task-covariance-preserving erasure is SPLINCE. The defensible contribution must therefore be
the **conditional-leakage subspace defined by task/domain score-Fisher conflict, gated by a
source-risk upper bound, with parameter-level conflict projection, applied under one budget
across layer/channel/node/edge specifically to cure the observed global-CMI collapse** — i.e.
items 1–4 above, on EEG. Until those land, this package is the *measurement + selection
scaffold*, honestly labelled.
