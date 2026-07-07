# FSR_37 — Recoverability Class Table (Phase 7A)

**Project FSR — Phase 7A.** Machine-auditable companion to FSR_36. Each row is a shortcut/corruption type; the
columns state, for a **deployable target-$X$-only batch-statistic (affine) operator**, what is observable, what is
invertible, what extra information a repair would require, the recoverability obstruction class (R0–R3, FSR_36 §2;
**non-exclusive** — see below), the current FSR evidence, and the **licensed action**. This is a characterization
grounded in the frozen controlled constructions, scoped to batch-affine operators, not a universal impossibility
theorem.

## Recoverability × information table
| Shortcut / corruption type | Observable from target-$X$? | Invertible from target-$X$ (batch-statistic operator)? | Extra info a repair needs | Class | Evidence | Licensed action |
|---|---|---|---|---|---|---|
| Deterministic mean offset (1st-moment) | yes (batch mean) | **yes** (mean alignment), up to conflation with genuine domain shift | none (target-$X$ only) | R0 | 4F (constant branch, scoped, construction-matched) | **repair** (E4), scoped + netted; disclose LODO fail |
| Fixed full-rank linear map (correlated w/ $z$) | yes (cross-cov) | **yes** (algebraic $A=(I+M)^{-1}$) | none | R0 | *none (algebraic, untested in FSR)* | repair *in principle*; unevidenced |
| Stochastic per-sample noise indep. of $z$ (2nd-moment) | yes (batch covariance) | **no by batch-affine op** (variance shrink ≠ per-sample inverse); even oracle direction fails | per-sample side info (non-affine denoiser out of scope) | R1 | 4G (none, even oracle) | **refuse / report**; flag as non-invertible by affine op |
| Task-entangled physiology (task-coupled subspace) | yes (subject subspace) | **no by erasure** ($S\not\perp T$: erasing hurts task) | task-preserving inverse (may not exist) | R2 | 4B (natural refuse), PC1 (oracle-subspace erase fails) | **refuse** blind erasure/repair |
| Prevalence-induced *learned* reliance | partly (representation shift) | **under-identified** (not a clean batch-deterministic offset) | labels / stress design / ≥3 datasets | R3 | PC2 (future; readiness = not eligible) | **flag / require stronger contract** |
| Concept shift ($P(Y\mid \text{concept})$) | **no** from $Z$-only | **no** | paired / minimal target labels (randomized contract) | R3 | sibling: concept-shift certification needs paired/min info | **certify only with paired info; else refuse** |
| Source-only checkpoint competence | weak (not a scalar) | **not scalar-invertible** | target info / support | R3 | sibling: source-side competence does not transport | **audit / refuse** (measurement→control gap) |

## How to read the table (obstructions are NON-exclusive)
R0–R3 are **not a partition**: they sit on different axes (moment/determinism, task-geometry, identifiability) and
can co-occur (e.g. $\varepsilon_i v$ with $v\in T$ is R1 **and** R2). When several hold, license the action by the
**binding (most-restrictive) obstruction** — refuse if any of {stochastic, task-coupled, under-identified} holds.
**R0 is the unique cell where none hold.**
- **Class R0 (recoverable):** affine-in-$z$ *and* task-separable *and* identified; a target-$X$-only affine
  operator inverts it (E4 for the constant/first-moment branch). Even here, recoverability $\ne$ generality — the
  4F repair is construction-matched (73% mechanical identity) and dataset-carried. The fixed-linear branch is
  algebraic and **untested** in FSR.
- **Class R1 (non-invertible by a batch-affine operator):** additive per-sample noise *independent of $z$*; no
  batch-affine operator recovers the per-sample realization (Prop 2, requires $\varepsilon\perp z$). A per-sample
  non-affine denoiser is *out of scope*, not excluded. A per-sample scalar *correlated with $z$* is affine-in-$z$
  and is R0, not R1.
- **Class R2 (task-coupled):** the nuisance subspace overlaps the task subspace; erasure destroys task (Prop 3).
  A valid repair needs a task-preserving inverse that may not exist.
- **Class R3 (under-identified from $X$):** identifiability itself fails from target-$X$; the shortcut can be
  *verified/refused* but repair requires a stronger information contract (labels, paired trials, randomized
  acquisition, or a controlled learned-reliance stress) — the boundary the sibling directions (concept-shift,
  source-only competence) independently hit.

## The single sentence
> A verified harmful shortcut becomes **repairable by a deployable target-$X$-only batch-affine operator** only if
> its corruption is **affine-in-$z$ and task-separable** and the **available information suffices to identify it**
> (R0). Per-sample stochastic (R1), task-coupled (R2), and target-$X$-under-identified (R3) shortcuts can be
> **detected, localized, and refused**, but not repaired by a target-$X$ batch-affine operator — they require
> stronger information contracts (per-sample side info, a task-preserving inverse, or labels/paired/randomized
> data). FSR's job is to place each shortcut against its binding obstruction and license the corresponding action
> (repair / refuse / require stronger information).

## Forbidden readings
- Treating the table as a proof of impossibility (R1/R2/R3 are non-invertible **by target-$X$-only batch-statistic
  operators**, not proven impossible under richer information).
- Presenting R0 (first-moment) repair as general/natural repair.
- Placing natural EEG spatial leakage anywhere but R2 (task-coupled) on current evidence (4B).
