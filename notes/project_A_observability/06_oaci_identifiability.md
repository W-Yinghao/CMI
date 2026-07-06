# Project A — OACI Identifiability

> The formal mathematical core. `01_information_regimes.md` is the axiom-layer draft; **this
> file is the citable theorem-definition layer** — the compatibility set `K_{R,C}`, the
> identified set, the `OA-0` definition, the non-identifiability certificate pattern, `MONO-1`
> and contract-strength monotonicity, the target-law-axiom caveat, and the estimation-layer
> separation. All later theorems use this proof language. Naming per `00_repo_audit.md §5`.

## 0. Purpose

Define **Observability-Aware Contracted Identifiability (OACI)** as the single formal layer
behind every Project A claim: *what is identifiable given an observation regime `R`, a declared
contract set `C`, and an analyst-chosen encoder `f`.* OACI identifies functionals of the world
`P` (and of `f`), never universal properties of `Z`.

## 1. World class

A **world** `P ∈ M` is a joint law over:
- domain split `S ∈ {src, tgt}`;
- domain factors `D = (site, subject, session, device, montage, rater, task-protocol, …)`;
- EEG observation `X`;
- latent task state `Y*`;
- observed label `Y`, via a label mechanism `p(Y | Y*, D)`;
- optional anchors `A`;
- optional **chosen representation** `Z = f(X)`.

> **Encoder-dependence.** `Z` is not primitive: it is `Z=f(X)` for an analyst-chosen `f`. OACI
> identifies functionals of `(P, f)`. A statement "`Z` has property Q" always means "`f(X)` has
> Q under `P`". (`01_information_regimes.md §1.1`.)

## 2. Observation operators

```
O_0(P) = source labeled law over (X, Y, D)                         [ S = src ]
O_1(P) = O_0(P)  ⊕  target unlabeled law over (X_T, D_T metadata)
O_2(P) = O_1(P)  ⊕  anchors A
```
with regimes `R0 = source-only`, `R1 = target-unlabeled`, `R2 = minimal-paired`. Each `O_R` is a
deterministic function of the world; `O_0` is a function of `O_1`, and `O_1` of `O_2` (drop the
extra coordinate) — the **refinement chain** `O_2 ⊒ O_1 ⊒ O_0` (§8).

## 3. Contracts as model-class restriction

A contract set `C ⊆ {C1,…,C12}` (`02_contract_taxonomy.md`) restricts the world class:
```
M(C) = { P ∈ M : P satisfies every contract in C }.
```
Contracts are **assumptions**, not observations, unless they are *checkable* in the regime
(i.e. `O_R` contains the coordinate the contract constrains — `02 §1`). `M(∅)=M`; enlarging `C`
shrinks `M(C)` (§9).

## 4. Compatibility set

For an observed value `o_R = O_R(P)`, the **compatibility set** is the set of worlds still
possible after observing `R` under `C`:
```
K_{R,C}(o_R) = { Q ∈ M(C) : O_R(Q) = o_R }.
```
`P ∈ K_{R,C}(O_R(P))` always. `K` is the central object: identifiability is *constancy of a
functional on `K`*.

## 5. `OA-0` — identifiability definition

> **`OA-0`.** A target functional `T` is **identifiable under `(R, C)`** iff
> ```
> ∀ P, Q ∈ M(C):   O_R(P) = O_R(Q)  ⟹  T(P) = T(Q),
> ```
> equivalently iff `T` is **constant on every** compatibility set `K_{R,C}(o_R)`.

## 6. Non-identifiability certificate pattern

> **Certificate.** `T` is **not** identifiable under `(R, C)` iff there exist `P, Q ∈ M(C)` with
> `O_R(P)=O_R(Q)` but `T(P) ≠ T(Q)` — two worlds in the **same** compatibility set that disagree
> on `T`.

This is the single proof pattern used throughout Project A:

| certificate | `(R, C)` | held equal | `T` disagreement |
|---|---|---|---|
| `TOS-1` / CE-R0-1/2/3 | R0 | source law | target risk / gain sign / prior |
| `TU-2` / CE-R1-1 | R1 | source + `p_T(X)` | target concept `p_T(Y\|X)` |
| CE-R1-2 | R1, ¬C3 | `p_T(z)` | target prior `π_T` |
| CE-C1-1 | R1, ¬C1 | source class-conditionals | mixture feasibility of `p_T(z)` |
| CE-MP-1 | R2, ¬C8 | one anchor `u↦v` | transform on un-anchored directions |
| CE-C11-1 | R2, ¬C11 | observed pair | true transport |
| CE-MONO-1 | R0 vs R1 | source law | target `X` marginal (regime-separating) |

## 7. Partial identification (identified set)

When `T` is not point-identifiable, OACI does not go silent — it reports the **identified set**:
```
I_{R,C}(o_R; T) = { T(Q) : Q ∈ K_{R,C}(o_R) }.
```
`T` is point-identifiable iff `I` is a singleton. **Any reported bound on `T` is valid only if it
covers `I_{R,C}`.** A method that reports a tighter interval than `I` is either using an
undeclared contract or overclaiming. (This is where source-only "risk bounds" must live: valid
only under an explicitly declared, `I`-covering contract — `03_… §6`.)

## 8. `MONO-1` — information monotonicity

> **`MONO-1`.** If `O_a ⊒ O_b` (`O_a` refines `O_b`), then for every `P`
> ```
> K_{a,C}(O_a(P)) ⊆ K_{b,C}(O_b(P)),
> ```
> hence: point-identifiability can only **increase**, identified sets `I` can only **shrink**,
> and bounds can only **tighten**, along `R0 ⊑ R1 ⊑ R2`.

*Proof.* If `Q ∈ K_{a,C}(O_a(P))` then `Q ∈ M(C)` and `O_a(Q)=O_a(P)`; refinement gives
`O_b(Q)=O_b(P)`, so `Q ∈ K_{b,C}(O_b(P))`. Thus `K_a ⊆ K_b`. A functional constant on `K_b` is
constant on the subset `K_a` (identifiability monotone); and `I_a = T(K_a) ⊆ T(K_b) = I_b`
(identified set shrinks). ∎

> **Non-interchangeability corollary.** `O_0` is the source-marginal of `O_1`; more *source*
> data refines only that coordinate and leaves the target coordinates of `O_1/O_2` fixed.
> Therefore a compatibility gap separated by a **target** coordinate cannot be shrunk by source
> breadth. **Certificate: CE-MONO-1** (`07_… §11`) — two worlds with identical source, `R0`-equal,
> `R1`-distinct via the target `X` marginal; adding identical source domains changes neither.

## 9. Contract-strength monotonicity

> **Contract-strength.** If `C_strong ⟹ C_weak` (every world satisfying `C_strong` satisfies
> `C_weak`), then `M(C_strong) ⊆ M(C_weak)` and `K_{R,C_strong} ⊆ K_{R,C_weak}`. Therefore, as
> assumptions strengthen:
> - the **class of point-identifiable functionals** can **expand**;
> - for a **fixed** functional `T`, the **identified set** `I_{R,C}(o_R; T)` can only **shrink**.

(Do not conflate these: the *set of functionals that become point-identifiable* grows, while the
*identified set of any one `T`* shrinks.) Crucially this is **assumption-strength, not
evidence.** Shrinking `M(C)` by assuming more can manufacture point-identifiability that the data
did not provide. OACI therefore always reports
*which* contracts are **checkable** in `R` (data-supported) vs **assumed** (declared but
unfalsifiable in `R`) — the audit's and `02`'s R0/R1/R2 checkability columns.

## 10. Target-law-axiom caveat

A contract that **fully fixes `P_T` as a function of `P_S`** (the extreme `P_T = P_S`) makes
target functionals computable — but this is a **declared target-law axiom**, not
source-only *identification*. OACI labels such a `C` explicitly as an axiom on the target law;
its conclusions inherit that axiom and may not be reported as "identified from `R0` observation"
(`03_tos_source_only_ceiling.md` Remark, cases (i) vs (ii)).

## 11. Estimation layer (kept separate)

OACI is a **population** theory. Three layers must not be conflated (`01_… §7`,
`02_… §3`):

| layer | question | governed by |
|---|---|---|
| identifiability | is `T` constant on `K_{R,C}`? | `OA-0` (this file) |
| finite-sample estimation | can `O_R` be estimated from a sample? | statistics (rates, clustered bootstrap) |
| neural-estimator fidelity | does `f`/critic realise the intended functional? | **C5** (`q_ψ→p(D\|Z,Y)`) |

`C5` is an estimator-fidelity contract, **not** part of `O_R`. A point-identified `T` can still
be mis-measured by a sub-optimal critic (P0-2); a non-identified `T` cannot be rescued by any
estimator, however good.

---

**Downstream.** `05_csc_shift_calculus.md` uses `K_{R,C}` / identified-set language to bound
what each *shift claim* may assert; `08_experimental_protocol.md` turns `OA-0` + the checkable/
assumed split into a per-experiment claim ledger.
