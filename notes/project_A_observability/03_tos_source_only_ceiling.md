# Project A — `TOS-1` Source-only Ceiling

> The flagship theorem. Strict source-only information cannot identify any target-specific
> functional. The proof is by **indistinguishable worlds** (`OA-0` contrapositive,
> `01_information_regimes.md` §5) and requires **no model training** — identifiability is a
> population property. §4 gives an exact binary certificate; `07_counterexample_catalog.md`
> and `counterexamples/run_counterexamples.py` execute it. Naming per `00_repo_audit.md` §5.

## 0. Claim

Under the source-only regime R0, no non-trivial functional of the target law is identifiable
without an external contract that ties the target to the source.

## 1. Regime

R0 observes only the source labeled law (`01_information_regimes.md` §2):
```
O_0(P) = P_S(X, Y, D)          [ source membership only ]
```
No target `X`. No target `Y`. No target prior. No target concept. No target transform.

Throughout, a **target functional** `T(P)` depends only on the target law `P_T` (e.g.
`R_T(h)`, `π_T`, `p_T(Y|X)`, the gain `Δ_T`, or `sign Δ_T`). A functional is **trivial** if it
is constant across the whole model class (then it carries no information and is vacuously
identifiable).

## 2. `TOS-1` theorem

> **`TOS-1` (source-only ceiling).** Let `C` be a **target-free-coordinate** contract set:
> `M(C)` factorizes over the source/target split, i.e.
> ```
> ∀ P_S ∈ proj_S(M(C)),  ∀ P_T ∈ proj_T(M(C)):  (P_S, P_T) ∈ M(C)
> ```
> (equivalently, `C` places no restriction linking the target law to the source law). Then
> every non-trivial target functional `T` is **non-identifiable under `(R0, C)`**. In
> particular the following are all non-identifiable under R0:
> - target risk `R_T(h)` of a fixed classifier `h`;
> - target prior `π_T(y)`;
> - target concept `p_T(Y | X)`;
> - target adaptation gain `Δ_T = bAcc_T(h_adapt) − bAcc_T(h_identity)` and even its **sign**;
> - target harm (`Δ_T < 0`).

The theorem is a direct instantiation of `OA-0`: because `O_0` reads nothing about the target,
the target law is a **free coordinate** of the world, and any functional that depends on it can
be moved without changing `O_0`.

## 3. Proof (indistinguishable worlds)

Fix any source law `P_S ∈ proj_S(M(C))`. By the factorization hypothesis, the `P_S`-slice
`{ P_T : (P_S, P_T) ∈ M(C) }` equals the full target projection `proj_T(M(C))`. Non-triviality
of `T` over `M(C)` gives two target laws `P_T ≠ Q_T` in `proj_T(M(C))` with `T(P_T) ≠ T(Q_T)`;
by factorization **both pair with the same `P_S`**.

Construct two worlds:
```
P = (P_S, P_T),      Q = (P_S, Q_T),      both in M(C).
```
They share the source law, so
```
O_0(P) = P_S = O_0(Q).
```
But `T(P) = T(P_T) ≠ T(Q_T) = T(Q)`. By the `OA-0` contrapositive, `T` is not identifiable
under `(R0, C)`. ∎

**Remark (why contracts are the only escape, and what "escape" means).** The proof breaks only
when `C` forces `P_T` to be a function of `P_S`. Two cases must be distinguished:

- **(i) Ordinary shared-mechanism contracts that still leave a target coordinate free** — e.g.
  C2 alone (`p_T(z|y)=p_S(z|y)`) with `π_T`, target concept, or transport still unconstrained —
  **do not** identify any target-specific functional under R0: there is no observed target
  coordinate for them to act on. They bite only once such a coordinate is *observed* — e.g.
  C2 **plus an observed `p_T(z)`** identifies `π_T`, but that is regime R1, not R0.
- **(ii) A contract that *fully fixes* `P_T` as a function of `P_S`** (the extreme `P_T=P_S`) can
  of course "yield" target quantities — but then the target law is an **external axiom**, not
  something *identified from R0 observations*. Project A records such assumptions as **declared
  target-law axioms**, never as source-only evidence.

Either way, target *claims* require target *observation* (R1/R2) or an explicitly-declared
target-law axiom — not target *assumptions* dressed as inference. This is the precise sense in
which R0 is a ceiling.

## 4. Binary exact counterexample (the `G / B` world pair)

A minimal, fully explicit instance. Let `X, Y ∈ {0,1}`.

**Source law (identical in both worlds).** `Y = X` deterministically on the source:
```
P_S(X=0, Y=0) = 1/2,     P_S(X=1, Y=1) = 1/2,     else 0.
```

**Target feature marginal (identical in both worlds).**
```
p_T(X=0) = p_T(X=1) = 1/2.
```

**Target concept (the only difference).**

| | world **G** (`Y=X`) | world **B** (`Y=1−X`) |
|---|---|---|
| `p_T(X=0, Y=0)` | 1/2 | 0 |
| `p_T(X=0, Y=1)` | 0 | 1/2 |
| `p_T(X=1, Y=0)` | 0 | 1/2 |
| `p_T(X=1, Y=1)` | 1/2 | 0 |

Both worlds share `P_S` and share `p_T(X)`. Therefore:
```
O_0(G) = O_0(B)        (same source law)                          → indistinguishable under R0
O_1(G) = O_1(B)        (same source law AND same target p_T(X))   → indistinguishable under R1 too
```
The second line is the bonus: **even target-unlabeled data (R1) cannot separate `G` from `B`**,
because they differ only in `p_T(Y|X)`. That is exactly `TU-2`
(`07_counterexample_catalog.md`, CE-R1-1).

### 4.1 Adaptation-gain sign is non-identifiable

Take the source-trained classifier `h_id(x)=x` (it is Bayes-optimal on the source, where
`Y=X`) and a candidate "adapted" classifier `h_ad(x)=1−x` (a label flip). With balanced target
classes, balanced accuracy equals accuracy:

| | `bAcc_T(h_id)` | `bAcc_T(h_ad)` | gain `Δ_T = bAcc(h_ad) − bAcc(h_id)` |
|---|---|---|---|
| world **G** | 1 | 0 | **−1** |
| world **B** | 0 | 1 | **+1** |

The two worlds are R0- (and R1-) indistinguishable, yet the sign of the adaptation gain flips.
So **no source-only rule can decide whether adapting helps or harms** — it would have to output
the same decision on `G` and `B`, but the correct decisions have opposite sign. This is the
identifiability core of the project's repeated empirical finding that source-only harm gates do
not transfer (the measurement→control gap; audit C9).

### 4.2 Target risk is non-identifiable

The same pair certifies target-risk non-identifiability directly. For the fixed classifier
`h_id(x)=x` under 0-1 loss:
```
R_T^G(h_id) = P_T^G(Y ≠ X) = 0,          R_T^B(h_id) = P_T^B(Y ≠ X) = 1.
```
Same `O_0`, different target risk of the *same* classifier → `R_T(h)` is not identifiable under
R0 (CE-R0-1).

## 5. Target prior is non-identifiable (CE-R0-3)

Even more directly, keep any source law and take two target priors
```
π_T^A = (0.2, 0.8),      π_T^B = (0.8, 0.2).
```
Since `O_0` observes nothing of the target, both are compatible with the same source, and
`π_T^A ≠ π_T^B`. Target prior is not identifiable under R0. (Contrast R1: `π_T` becomes
identifiable *only* under C2∧C1∧C3 — theorem `TU-1`.)

## 6. What source-only **can** still identify

`TOS-1` is a ceiling on *target* functionals, not a nihilism. R0 identifies:
- source risk `R_S(h)` and source-internal LOSO robustness;
- source leakage diagnostics `I(Z_s;D|Y)`, `I(Z_s;D)`, `I(Y;D|Z_s)` (for a fixed `f`, up to the
  C5/C6 estimator caveats);
- a source-domain safety *proxy* (the inner-LOSO gain distribution) — as a **source** object;
- source-side risk **bounds** on the target *only when* an external, separately-justified
  contract is declared (e.g. a bounded-transport C8 assumption gives a target-risk envelope, not
  a point estimate).

## 7. What source-only must **not** claim

No source-only method may claim, without additional target information (R1/R2) or a contract
linking `P_T` to `P_S`:
- target-specific adaptation-gain identification;
- target harm certification;
- target prior estimation;
- target concept-shift detection;
- target TTA safety certification.

Each such claim is refuted by a `G/B`-style world pair. Project A's role is to force every
adaptation claim to declare its regime and contract, and to attach the indistinguishable-worlds
certificate when the claim exceeds what the regime identifies.

## 8. Reproduction

The proof needs no training. The exact-discrete certificates of §4–§5 are executed by
```
conda run -n icml python notes/project_A_observability/counterexamples/run_counterexamples.py
```
which prints `source_equal`, the per-world target risk, the gain-sign flip, and the R1 concept
checks, and asserts each expected (in)equality (fails loud on any arithmetic error). A
simulator-backed *illustration* on `h2cmi.data.eeg_simulator` (matched target `X`, divergent
label mechanism) is included there as illustration only — the mathematics above is the proof.
