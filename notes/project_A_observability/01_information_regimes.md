# Project A — Information Regimes

> The axiom layer. Defines the observation operators for **R0 source-only**, **R1
> target-unlabeled**, **R2 minimal-paired**, states the OACI identifiability definition
> (`OA-0`) and information monotonicity (`MONO-1`), and fixes the estimation-vs-identification
> separation the rest of Project A relies on. Naming follows §5 of `00_repo_audit.md`
> (`OA-0 / TOS-1 / TU-1 / TU-2 / MP-1 / PD-1 / MONO-1`; legacy chain-rule identity = `ID-1`).

## 0. Purpose

Every later theorem cites this file for three things:

1. **What is observed** in each regime — the operator `O_R(P)`.
2. **What "identifiable" means** — `OA-0`.
3. **Why more information helps but is not interchangeable** — `MONO-1`.

The point of Project A is that adaptation methods routinely *estimate* quantities that are not
*identified* by their regime. Identifiability is a property of the population + the observation
operator; it precedes any estimator. This file makes that operator explicit.

## 1. World model

A **world** is a joint distribution `P` over:

| symbol | meaning |
|---|---|
| `S ∈ {src, tgt}` | source / target membership indicator |
| `D = (site, subject, session, device/montage, rater, …)` | domain factors (the DAG of `domains/dag.py`) |
| `X` | the EEG observation (raw or fixed-preprocessed trial) |
| `Y* ` | latent task state (the "true" class) |
| `Y` | observed label; related to `Y*` by a **label mechanism** `p(Y | Y*, D)` (rater/site channel) |
| `A` | optional anchor variables (paired sessions, montage maps, calibration pairs, resting-state) |

The model class `M(C)` is the set of worlds consistent with a declared contract set `C`
(the C1–C10 of `00_repo_audit.md`).

### 1.1 `Z = f(X)` is chosen, not primitive — the critical caveat

`Z` is **not** a variable of the world. It is `Z = f(X)` for an analyst/model-chosen encoder
`f`. Consequences that the audit's regime matrix must be read through:

- The primitive observables are always `X, Y, D, A`. **Any statement about `Z` is conditional
  on a fixed `f`.** "Observed `p_S(z|y)`" is shorthand for "`p_S(f(X)|y)` for the analyst's
  chosen `f`", not a regime primitive.
- Two encoders `f, f'` induce different `Z` and therefore different leakage values
  `I(Z;D|Y)`; identifiability statements about `X`-level quantities are encoder-free, statements
  about `Z`-level quantities carry an implicit "for this `f`".
- This is why **critic sufficiency (C5)** and the `ID-1` accounting live at the *estimator*
  layer, not the observation operator (see §7): the map `X → Z` and the map `Z → Î(Z;D|Y)` are
  both analyst choices with their own fidelity gaps.

Convention below: observation operators are stated over `(X, Y, D, A)`; where we write `p(z|y)`
we mean "for the fixed encoder under discussion".

## 2. R0 — source-only

**Observed.**
```
O_0(P) = the source labeled law/sample over (X_s, Y_s, D_s)         [ S = src only ]
```
i.e. the joint distribution of source trials, their observed labels, and their domain-factor
levels. No target quantity of any kind is in `O_0`.

**Not observed.** target `X_T`, target `Y_T`, target prior `π_T(y)`, target concept
`p_T(Y|X)`, target label mechanism, target acquisition transform.

**Identifiable (encoder-free or for a fixed source-trained `f`).**
- source risk `R_S(h)` of any fixed `h`;
- source priors `π_d(y) = p(y|D=d)` and `p_S(y)`;
- source leakage diagnostics `I(Z_s;D|Y)`, `I(Z_s;D)`, `I(Y;D|Z_s)` (for the fixed `f`, up to
  C5/C6 estimator caveats);
- the source inner-LOSO distribution of adaptation gain `Δ = bAcc_adapt − bAcc_identity`.

**Not identifiable.** target risk, target prior, target concept, target adaptation gain, target
harm sign. These are the subject of `TOS-1` (`03_tos_source_only_ceiling.md`).

## 3. R1 — target-unlabeled

**Observed.**
```
O_1(P) = O_0(P)  ⊕  the unlabeled target law/sample over X_T  ( + target D metadata if present )
```

**Newly identifiable (over R0).** the target **feature marginal** `p_T(X)` (hence `p_T(Z)` for
a fixed `f`).

**Identifiable only under contract.** the target prior `π_T(y)` — **only** under
**C2 ∧ C1 ∧ C3** (shared class-conditional geometry, support overlap, full-rank confusion /
mixture separability), via the mixture identity `p_T(z) = Σ_y π_T(y) p_ref(z|y)` equivalently
`w = C⁻¹μ`. This is theorem `TU-1`.

**Still not identifiable.** target `p_T(Y|X)` (concept), target accuracy, target adaptation
gain, whether adaptation helped. That target-unlabeled data cannot identify concept is theorem
`TU-2` — the same `p_T(X)` is compatible with many `p_T(Y|X)`
(`07_counterexample_catalog.md`, CE-R1-1).

## 4. R2 — minimal-paired

**Observed.**
```
O_2(P) = O_1(P)  ⊕  anchors A
```

**Anchor types.**
- **label anchors** — a few `(X_T, Y_T)` pairs;
- **transport anchors** — paired source↔target sessions, or pre→post calibration pairs;
- **montage anchors** — electrode/device/canonicalisation correspondences;
- **resting-state anchors** — subject physiology without task labels.

**Identifiable.** target risk on the labeled slice (directly).

**Identifiable only under contract.** low-dimensional acquisition/montage transport (theorem
`MP-1`, under **C8** — low-dim, invertible, full-rank, overlap); a bounded target risk;
label-mechanism residual on the anchored subset.

**Still not identifiable.** unconstrained high-dimensional transport, or the full concept
mechanism, from insufficient anchors.

## 5. `OA-0` — OACI identifiability definition

Let `T` be a target functional (e.g. `T(P) = R_T(h)`, or `T(P) = π_T`, or `T(P) = sign Δ_T`).

> **`OA-0` (identifiability under an observation operator and a contract).**
> `T` is **identifiable under `(R, C)`** iff
> ```
> ∀ P, Q ∈ M(C):    O_R(P) = O_R(Q)   ⟹   T(P) = T(Q).
> ```

Contrapositive (the **certificate of non-identifiability**, used everywhere in Project A): `T`
is **not** identifiable under `(R, C)` iff there exist two worlds `P, Q ∈ M(C)` with
`O_R(P) = O_R(Q)` but `T(P) ≠ T(Q)`. Such a `(P, Q)` pair is an **indistinguishable-worlds
counterexample**. Every non-identifiability claim in this project must ship one
(`07_counterexample_catalog.md`).

Note `OA-0` is relative to a contract set `C`: enlarging `C` (assuming more) shrinks `M(C)` and
can only *increase* the set of identifiable functionals. This is the lever adaptation methods
pull — and the thing they must declare.

## 6. `MONO-1` — information monotonicity

Define the **compatibility set** of a world `P` under `(R, C)`:
```
K(P; R, C) = { Q ∈ M(C) : O_R(Q) = O_R(P) }.
```
`T` is identifiable at `P` iff `T` is constant on `K(P; R, C)`.

> **`MONO-1`.** If observation operator `O_a` **refines** `O_b` (i.e. `O_a(P)=O_a(Q) ⟹
> O_b(P)=O_b(Q)`; written `O_a ⊒ O_b`), then for every `P`:
> ```
> K(P; O_a, C) ⊆ K(P; O_b, C).
> ```
> Hence the set of `(R,C)`-identifiable functionals is monotone non-decreasing along
> `R0 ⊑ R1 ⊑ R2`, and any bound derived from `O_a` is at least as tight as one from `O_b`.

*Proof.* `O_0` is a function of `O_1` (drop target `X_T`), and `O_1` is a function of `O_2`
(drop anchors `A`); so `O_2 ⊒ O_1 ⊒ O_0`. If `Q ∈ K(P;O_a,C)` then `O_a(Q)=O_a(P)`, and
refinement gives `O_b(Q)=O_b(P)`, so `Q ∈ K(P;O_b,C)`. A functional constant on the larger set
`K(P;O_b,C)` is constant on the subset `K(P;O_a,C)`; the converse fails in general. ∎

> **`MONO-1` corollary (non-interchangeability).** More *source* data refines the **source
> marginal** of `O_0` but leaves the **target** coordinates of `O_1/O_2` untouched. Therefore
> source breadth cannot shrink a `K` gap that is separated only by a target coordinate. A
> `TOS-1` world pair (`03_…`) that differs only in the target law is invariant to adding source
> subjects — so "collect more source subjects" is provably not a substitute for target-unlabeled
> `X_T` or paired anchors `A`.

## 7. Estimation vs identification (three distinct layers)

Keep these separate; conflating them is red-flag #11 of the audit.

| layer | question | governed by |
|---|---|---|
| **population identifiability** | is `T(P)` fixed by `O_R(P)` under `C`? | `OA-0` (this file) |
| **finite-sample estimation** | can `T` be estimated from a finite sample of `O_R(P)`? | statistics (rates, CIs, clustered bootstrap) |
| **neural-estimator fidelity** | does the chosen `f` / critic `q_ψ` realise the intended functional? | **C5** critic sufficiency; the `X→Z→Î` maps |

`C5` (Step-A convergence, `q_ψ → p_θ(D|Z,Y)`) is an **estimator-fidelity** contract, *not* part
of the observation operator: even a perfectly identified population quantity can be mis-measured
by a sub-optimal critic (the audit's `L_CMI≈0` while true CMI>0). Project A reports
identifiability and fidelity separately, and never lets a low measured leakage stand in for an
identified target guarantee.

---

**Downstream.** `03_tos_source_only_ceiling.md` instantiates `OA-0` at `R=R0` to prove `TOS-1`;
`07_counterexample_catalog.md` supplies the indistinguishable-worlds pairs (`OA-0`
contrapositive) for `TOS-1`, `TU-2`, and the `TU-1`/C3 failure.
