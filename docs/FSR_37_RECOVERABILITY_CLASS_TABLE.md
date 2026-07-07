# FSR_37 — Recoverability Class Table (Phase 7A)

**Project FSR — Phase 7A.** Machine-auditable companion to FSR_36. Each row is a shortcut/corruption type; the
columns state, for a **deployable target-$X$-only** setting, what is observable, what is invertible, what extra
information a repair would require, the recoverability class (R0–R3, FSR_36 §2), the current FSR evidence, and the
**licensed action**. This is a characterization grounded in the frozen controlled constructions, not a universal
theorem.

## Recoverability × information table
| Shortcut / corruption type | Observable from target-$X$? | Invertible from target-$X$ (batch-statistic operator)? | Extra info a repair needs | Class | Evidence | Licensed action |
|---|---|---|---|---|---|---|
| Deterministic mean offset (1st-moment) | yes (batch mean) | **yes** (mean alignment), up to conflation with genuine domain shift | none (target-$X$ only) | R0 | 4F (scoped, construction-matched) | **repair** (E4), scoped + netted; disclose LODO fail |
| Stochastic per-sample perturbation (2nd-moment) | yes (batch covariance) | **no** (variance shrink ≠ per-sample inverse); even oracle direction fails | per-sample side info | R1 | 4G (none, even oracle) | **refuse / report**; flag as non-invertible |
| Task-entangled physiology (task-coupled subspace) | yes (subject subspace) | **no by erasure** ($S\not\perp T$: erasing hurts task) | task-preserving inverse (may not exist) | R2 | 4B (natural refuse), PC1 (oracle-subspace erase fails) | **refuse** blind erasure/repair |
| Prevalence-induced *learned* reliance | partly (representation shift) | **under-identified** (not a clean batch-deterministic offset) | labels / stress design / ≥3 datasets | R3 | PC2 (future; readiness = not eligible) | **flag / require stronger contract** |
| Concept shift ($P(Y\mid \text{concept})$) | **no** from $Z$-only | **no** | paired / minimal target labels (randomized contract) | R3 | sibling: concept-shift certification needs paired/min info | **certify only with paired info; else refuse** |
| Source-only checkpoint competence | weak (not a scalar) | **not scalar-invertible** | target info / support | R3 | sibling: source-side competence does not transport | **audit / refuse** (measurement→control gap) |

## How to read the table
- **Class R0 (recoverable):** the corruption is batch-deterministic *and* task-separable; a target-$X$-only affine
  operator inverts it (E4 for the first moment). Even here, recoverability $\ne$ generality — the 4F repair is
  construction-matched (73% mechanical identity) and dataset-carried.
- **Class R1 (non-invertible from batch moments):** per-sample stochastic corruption; no batch-statistic operator
  recovers the per-sample realization (Prop 2). Repair would need per-sample side information.
- **Class R2 (task-coupled):** the nuisance subspace overlaps the task subspace; erasure destroys task (Prop 3).
  A valid repair needs a task-preserving inverse that may not exist.
- **Class R3 (under-identified from $X$):** identifiability itself fails from target-$X$; the shortcut can be
  *verified/refused* but repair requires a stronger information contract (labels, paired trials, randomized
  acquisition, or a controlled learned-reliance stress) — the boundary the sibling directions (concept-shift,
  source-only competence) independently hit.

## The single sentence
> A verified harmful shortcut becomes **repairable** only if its corruption is **batch-deterministic and
> task-separable** and the **available information suffices to identify it** (R0). Per-sample stochastic (R1),
> task-coupled (R2), and target-$X$-under-identified (R3) shortcuts can be **detected, localized, and refused**,
> but not repaired by a deployable target-unlabeled operator — they require stronger information contracts. FSR's
> job is to place each shortcut in its class and license the corresponding action (repair / refuse / require
> stronger information).

## Forbidden readings
- Treating the table as a proof of impossibility (R1/R2/R3 are non-invertible **by target-$X$-only batch-statistic
  operators**, not proven impossible under richer information).
- Presenting R0 (first-moment) repair as general/natural repair.
- Placing natural EEG spatial leakage anywhere but R2 (task-coupled) on current evidence (4B).
