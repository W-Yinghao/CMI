# Project A — Contract Taxonomy

> The **canonical information-contract registry**. This file supersedes the first-pass contract
> sketch in `00_repo_audit.md §4`; where the two differ, **this file governs** (crosswalk in
> §7). Naming follows `00_repo_audit.md §5` (`OA-0 / TOS-1 / TU-1 / TU-2 / MP-1 / PD-1 / MONO-1`;
> legacy chain-rule identity = `ID-1`). Regimes and `OA-0` are defined in
> `01_information_regimes.md`.

## 0. Purpose

Project A separates four things that adaptation papers routinely conflate:

- **observed information** — the operator `O_R` (`01_information_regimes.md`);
- **declared contracts** — the assumptions `C` a method depends on (this file);
- **identifiable estimands** — what `(R, C)` pins down under `OA-0`;
- **estimator fidelity** — whether the chosen `f` / critic realises the intended functional.

> **Admissibility rule.** A method may claim an estimand `T` **only if** its observed regime
> `R` plus its *declared* contracts `C` identify `T` under `OA-0`. Every claim must name `R`,
> `C`, and — if `T` exceeds what `(R,C)` identifies — attach the indistinguishable-worlds
> certificate (`07_counterexample_catalog.md`).

## 1. Contract record schema

Each contract is recorded as:

- **ID / Name**
- **Mathematical statement**
- **Supports which theorem / estimand**
- **Needed in which regime**
- **Checkable from R0? R1? R2?**
- **Failure certificate** (the counterexample or corrected-claim that fires when it breaks)
- **Existing code hook**
- **Overclaim to avoid**

The distinction that matters most: a contract is **checkable** in a regime iff the regime's
observation operator contains the coordinate the contract constrains. Checkability ≠ truth — a
contract can be checkable and false, or true and uncheckable.

## 2. Contract table

| ID | Name | Statement (informal) | Supports | R0 check | R1 check | R2 check | Failure certificate |
|---|---|---|---|---|---|---|---|
| **C1** | class support overlap | `supp p_T(z\|y) ⊆ supp p_S(z\|y)` ∀y | TU-1, MP-1, risk transfer | ✗ | partial (target Z mass vs source support) | ✓ on paired slice | **CE-C1-1** (target mass off source support) |
| **C2** | shared class-conditional geometry | `p_T(z\|y)=p_S(z\|y)=p_ref(z\|y)` ∀y | TU-1, PD-1 residual reading | ✗ | ✗ (≡ prior/concept change) | ✓ with anchors | **CE-R1-1** |
| **C3** | mixture / full-rank identifiability | `B_{z,y}=p_ref(z\|y)` has full **column** rank (=\|Y\|); confusion `C` invertible | **TU-1** | ✓ (source `B`/`C` rank observable) | ✓ | ✓ | **CE-R1-2** |
| **C4** | stable label mechanism / no target concept shift | `p_T(Y\|Y*,D)=p_S(Y\|Y*,D)` and `p_T(Y*\|X)=p_S(Y*\|X)` | target-risk transfer, concept reading | ✗ | ✗ | partial (few target labels test it) | **CE-R1-1** |
| **C5** | critic / estimator sufficiency | `q_ψ → p_θ(D\|Z,Y)` (Step-A converged) — **estimator layer, not observability** | measured leakage = population leakage | diagnosable only (bounded) | diagnosable only | diagnosable only | **P0-2** (`q_ψ=π_y ⇒ R=0` while CMI>0) |
| **C6** | representation sufficiency / predictive span | `I(Y;X)=I(Y;Z)` (Z sufficient for Y); **precond.** each domain spans ≥2 classes | interpreting `I(Y;D\|Z)` as concept, not Z-loss | partial (source `I(Y;Z)` vs strong ref; span checkable ✓) | partial | partial | **P0-4** (Z-insufficiency inflates `I(Y;D\|Z)`) |
| **C7** | reference-prior / GLS reweighting availability | per-domain `π_d(y)=p(y\|D=d)` known; `w_d(y)=π*(y)/π_d(y)` ⇒ `Ĩ(Y;D)=0` | **PD-1** | **source-side ✓; target-side ✗** | target prior only via **TU-1** (C1∧C2∧C3) | ✓ with anchors | **CE-R1-2** |
| **C8** | low-dimensional invertible transport | transform in low-dim invertible family (near-identity affine, `‖A−I‖` small), full-rank, overlap | **MP-1** | ✗ | weak (marginal shift visible, transform not) | ✓ with paired anchors | **CE-MP-1** (transform not unique; too few anchors) |
| **C9** | source-to-target safety transfer | inner-LOSO source gain dist. `Δ` transfers to unseen targets | safety gate | source proxy only | weak | calibratable with target labels | **CE-R0-2 / TOS-1** (gain-sign flip) |
| **C10** | zero-Bayes / `D⊥Y\|Z` escape | `H(Y\|Z)=0` **sufficient not necessary**; correct: `I(Z;D\|Y)=0 ⇒` both CMIs 0 **iff** `I(Z;D)=I(Y;D)` ⇔ `D⊥Y\|Z` | legacy escape only | source diagnostic (`I(Z;D)` vs `I(Y;D)`, `D⊥Y\|Z` test) | source diagnostic persists; target-side ✗ | anchored | **P0-3** counterexample |
| **C11** | anchor validity | anchors pair samples from the **same** latent event / subject / device mechanism (no fake pairing, no label leakage) | **MP-1** | ✗ | ✗ | partial (anchor consistency checks) | **CE-C11-1** (same observed pair, different true transport) |
| **C12** | domain-factor separability | each `D_j` validly assigned acquisition (invariance-eligible) vs label-mechanism role; `determines_label` correct | OACI / CSC, legitimacy of invariance | metadata + test `D_j⇒Y` | source `D_j⇒Y` test (as R0) + target metadata | better with anchors | **P0-4** (subject=label `Y=g(D)`) |

## 3. Contracts that need the most care

### C7 — the source-side/target-side asymmetry (do not overclaim)
`C7` is **available source-side**: source labels give every source `π_d(y)`, so GLS reweighting
of the **source** law to a reference prior `π*` is a legitimate source-only operation, and
`PD-1` (`04_prior_decoupled_theory.md`) is a source-side accounting result. **`C7` is NOT
available target-side under R0**: the target prior `π_T` is unobserved, so *target* reweighting
/ *target-prior* estimation is **not** a source-only operation. It becomes available only under
R1 **and** `TU-1` (C1∧C2∧C3), and even then it is an *estimate*, not a free lunch. Writing "GLS
gives us the target prior" as if source-only is the single most common overclaim this registry
exists to block.

### C4 vs C6 vs C2 — three different "concept" preconditions
- **C2** (shared class-conditional *geometry*, `p_T(z|y)=p_S(z|y)`) is about the **feature**
  channel; its failure is a covariate/geometry mismatch.
- **C4** (stable **label mechanism** / no concept shift, `p_T(Y|X)=p_S(Y|X)`) is about the
  **label** channel; its failure is a true change in `p_T(Y|X)` (concept shift proper) — which,
  per P0-4, is exactly what `I(Y;D|Z)` alone cannot certify without C2∧C5∧C6.
- **C6** (representation **sufficiency**) is about the **encoder**: even with C2 and C4 true, a
  lossy `Z` inflates `I(Y;D|Z)` (P0-4). Only under **C2 ∧ C4 ∧ C6 ∧ C5** is a positive
  `I(Y;D|Z)` *evidence* of concept shift rather than an artifact.

### C5 is not an observability contract
`C5` (critic sufficiency) governs whether the *estimator* approximates the *population*
functional. It is listed here for completeness but belongs to the estimator-fidelity layer
(`01_information_regimes.md §7`), not the observation operator. A perfectly identified quantity
can still be mis-measured by a sub-optimal critic.

## 4. Regime-specific contract status

### R0 source-only
Contracts are checkable **only on source coordinates**. They **cannot certify target-specific
quantities** unless they fully fix `P_T` as an external axiom (in which case the target law is a
declared axiom, not source-only evidence — `03_tos_source_only_ceiling.md` Remark).
Source-checkable: C3, C5 (diagnostic), C6 (span + source sufficiency), C7 (source-side), C9
(source proxy), C10 (source diagnostic), C12 (metadata + `D_j⇒Y` test). Not source-checkable:
C1, C2, C4, C8, C11.

### R1 target-unlabeled
Adds observed target `X_T` ⇒ `p_T(z)`. Now checkable: target support/marginal shift (C1 partial).
**Identifies `π_T` only under C1 ∧ C2 ∧ C3** (theorem `TU-1`). **Cannot** check target concept
(C4) — `p_T(z)` is compatible with many `p_T(Y|X)` (`CE-R1-1`, theorem `TU-2`).

### R2 minimal-paired
Adds anchors. Can partially check the target label mechanism (C4 with a few labels), transport
(C8 with paired sessions), and anchor validity (C11). Still cannot identify unconstrained
high-dimensional mechanisms (C8 failure) or a full concept map without sufficient anchors.

## 5. Failure-certificate map

Every contract fires a specific, executable-or-cited certificate when it breaks:

| Broken contract | Certificate | Where |
|---|---|---|
| C2 / C4 (concept from unlabeled target) | **CE-R1-1** (= theorem `TU-2`) | `07_…` §4, `run_counterexamples.py` |
| C3 / C7-target (mixture rank failure) | **CE-R1-2** | `07_…` §5, `run_counterexamples.py` |
| C9 (source→target safety transfer) | **CE-R0-2** (gain-sign flip) / `TOS-1` | `07_…` §2, `03_…` §4.1 |
| C10 (zero-Bayes overclaim) | **P0-3** counterexample (`D=A,Y=(A,B),Z=A`) | `h2cmi/THEORY.md` |
| C5 (estimator overclaim) | **P0-2** (posterior-KL is not an upper bound) | `h2cmi/THEORY.md` |
| C6 (Z-insufficiency) | **P0-4** (`I(Y;D\|Z)` is a predictive-insufficiency diagnostic) | `h2cmi/THEORY.md` |
| C12 (subject=label / `determines_label` degeneracy) | **P0-4** (clinical degeneracy `D=subject⇒Y=g(D) ⇒ I(Y;D\|Z)=H(Y\|Z)`) | `h2cmi/THEORY.md` |
| C1 (support overlap) | **CE-C1-1** (target mass off source support ⇒ mixture infeasible) | `07_…` §8, `run_counterexamples.py` |
| C8 (high-dim transport) | **CE-MP-1** (transform not unique under too few anchors) | `07_…` §9, `run_counterexamples.py` |
| C11 (fake pairing) | **CE-C11-1** (same observed pair, different true transport) | `07_…` §10, `run_counterexamples.py` |

## 6. Contract-vs-estimator, and the declaration template

`C5` is the only entry that is an **estimator-fidelity** contract, not an observability
contract; keep it separate when reporting (a low measured leakage under a weak critic is not an
identified low leakage).

Every future Project A method / experiment section must fill this template:

```
Regime:                R0 | R1 | R2
Observed:              (the coordinates of O_R actually used)
Contracts invoked:     (subset of C1..C12, with which are checkable in this regime)
Identifiable estimand: (the T that (R, C) pins down under OA-0)
Estimator used:        (f / critic; note C5 fidelity diagnostics)
Failure mode:          (which certificate fires if a contract breaks)
```

## 7. Reconciliation with `00_repo_audit.md §4` (crosswalk)

The audit's §4 was a first-pass sketch (C1–C10). This registry (C1–C12) is canonical. Identical
entries: **C1, C2, C3, C5, C7, C8, C9, C10**. Two labels changed meaning, and two contracts are
new:

| audit §4 label | canonical label here | note |
|---|---|---|
| C4 (acquisition-vs-label-mechanism factor separation) | **C12** (domain-factor separability) | same concept, renumbered |
| C6 (domain-class span / decoder validity) | **C6** precondition (folded into representation/predictive validity) | span retained as C6's data-coverage precondition |
| — | **C4** (stable label mechanism / no target concept shift) | **new** — the label-channel contract |
| — | **C6** (representation sufficiency, `I(Y;X)=I(Y;Z)`) | **new** — the encoder contract (P0-4) |
| — | **C11** (anchor validity) | **new** — R2 pairing contract |

A one-line pointer to this crosswalk is added at the top of `00_repo_audit.md §4`; the audit's
prose C-references are not renumbered (to avoid churn), so read any audit C4/C6 through this
table.
