# Project A — CSC: Contracted Shift Calculus

> A calculus, not a taxonomy. It maps every **shift claim → required observation + contracts →
> identifiable estimand → failure certificate**, so the word "domain shift" can never hide an
> incompatible mechanism or an unearned target claim. Uses the `K_{R,C}` / identified-set
> language of `06_oaci_identifiability.md` and the contracts of `02_contract_taxonomy.md`.
> Naming per `00_repo_audit.md §5`.

## 0. Purpose

A shift claim is **admissible** only when it declares: the observed regime `R`, the *shifted
object*, the contracts used, the identifiable estimand, and the failure certificate that fires
if a contract breaks. CSC is the grammar that enforces this.

## 1. Factorized world view

Work over the full factorization
```
P(X, Y, Y*, D, A),   D = (site, subject, session, device, montage, rater, task-protocol).
```
**Do not collapse `D` into a single adversarial "domain" variable.** Different factors carry
different roles (acquisition vs label-mechanism — C12) and are checkable in different regimes;
collapsing them is how "domain shift" hides an incompatible mechanism.

## 2. Shift objects

| tag | name | changes | note |
|---|---|---|---|
| **S-acq** | acquisition / covariance / device / montage | `p(X\|Y*,D)` (hence `p(Z\|Y,D)`) | the *correctable* geometry shift |
| **S-prior** | label-prior shift | `p(Y\|D)` | removed by GLS reweighting (source-side) |
| **S-concept** | label-mechanism / task-state mechanism | `p(Y\|X,D)` or `p(Y\|Y*,D)` | the shift pooled alignment cannot fix |
| **S-support** | target support / overlap failure | target mass outside source support | breaks C1 |
| **S-transport** | low-dim paired transform | source↔target observation map | needs anchors (C8, C11) |
| **S-rater** | label protocol / rater / annotation | `p(Y\|Y*,D)` on the label channel | a *label-mechanism* factor, not encoder-invariant |
| **S-factor** | domain-factor degeneracy | `Y = g(D)` (e.g. subject determines label) | the SCPS/P0-4 degeneracy (C12) |

## 3. Contract map (shift → regime → contracts → estimand → certificate)

| Shift claim | Minimal regime | Required contracts | Identifiable object | Failure certificate |
|---|---|---|---|---|
| target **prior** shift | R1 | C1 ∧ C2 ∧ C3 | `π_T` (theorem `TU-1`) | **CE-R1-2** |
| target **concept** shift | R2 | C4 ∧ C6 ∧ C5 (+ anchors/labels) | a **bounded / tested** residual, not a point value | **CE-R1-1** (loss of R2 anchors → R1 non-ID); anchor rank/validity failure → **CE-MP-1 / CE-C11-1** |
| **support** failure | R1 | marginal support: none; class support: labels | off-source target mass | **CE-C1-1** |
| montage / **transport** | R2 | C8 ∧ C11 | transform, or a bound | **CE-MP-1 / CE-C11-1** |
| source **safety** transfer | R0 **+ target-law axiom** | C9 | source proxy only (unless the axiom is declared) | **CE-R0-2** |

Read the table with `06 §7`: where the "identifiable object" is a bound, it must cover the
identified set `I_{R,C}`; where it needs a target-law axiom (safety transfer), that axiom is a
declared assumption, not source-only evidence (`06 §10`).

## 4. CSC rules

- **CSC-R1.** A target *marginal* difference is **not** a concept-shift certificate
  (`p_T(X)` is compatible with many `p_T(Y|X)` — `TU-2` / CE-R1-1).
- **CSC-R2.** A small domain-classifier leakage is **not** a target-risk guarantee
  (leakage is a source-side diagnostic; measurement→control gap — C9).
- **CSC-R3.** Prior shift must be **decoupled** (GLS reweighting, `PD-1`) before any conditional
  leakage number is interpreted.
- **CSC-R4.** Class-conditional alignment requires **labels**, pseudo-labels *under a declared
  contract*, or anchors — never the raw target marginal alone.
- **CSC-R5.** Paired anchors identify **only the transform family they excite**; under-ranked or
  invalid anchors give non-identifiability (CE-MP-1 / CE-C11-1).
- **CSC-R6.** Domain factors must be **typed**: invariance over an *acquisition* factor
  (site/montage/session) is not equivalent to invariance over a *label-mechanism* factor
  (rater/diagnostic-site) — C12 / P0-4.

## 5. Regime-specific shift observability

- **R0 (source-only).** Can observe source-internal shift only. **Cannot** observe any target
  shift. Any target shift claim needs R1/R2 or a declared target-law axiom (`06 §10`).
- **R1 (target-unlabeled).** Can observe target **marginal / support** shift (S-acq marginal,
  S-support). **Cannot** observe target **concept** (S-concept). Identifies the target **prior**
  only under C1 ∧ C2 ∧ C3 (`TU-1`).
- **R2 (minimal-paired).** Can **test / bound** concept, transport, and anchor validity —
  depending on anchor *type* and *rank* (C8, C11); never identify an unconstrained high-dim
  mechanism.

## 6. Diagnostic vocabulary

**Use** (regime/contract-honest):
- "marginal feature shift observed" (R1);
- "target prior identifiable under `TU-1` contracts (C1∧C2∧C3)" (R1);
- "predictive-insufficiency residual `I(Y;D|Z)`" (a P0-4 diagnostic, not a shift label);
- "source-side leakage diagnostic `I(Z;D|Y)`" (R0, source only);
- "transport identifiable under C8 ∧ C11" (R2).

**Avoid** (forbidden overclaims — these must never appear as assertions):
- "target concept shift detected from unlabeled target" — forbidden (violates CSC-R1 / `TU-2`);
- "source-only target safety certified" — forbidden (violates C9 / `TOS-1`);
- "CMI guarantees accuracy" — forbidden (invariance ≠ accuracy);
- "GLS gives the target prior source-only" — forbidden (C7 asymmetry; needs R1 + `TU-1`).

> `I(Y;D|Z)` is **diagnostic vocabulary, not a shift label.** It enters a *concept*
> interpretation only under C4 ∧ C6 ∧ C5 (+ anchors/labels), and even then as a bounded/tested
> residual (§3), never as a source- or R1-only "concept detected".

## 7. Relation to EEG (shift objects → mechanisms)

- **S-acq / S-transport** ← subject anatomy / skull conductivity, electrode impedance & montage,
  device sampling / filter response;
- **S-prior** ← class-prevalence differences across sites/cohorts;
- **S-concept** ← task strategy, clinical phenotype, task-protocol change;
- **S-rater** ← rater / diagnostic-site / annotation-protocol differences (label channel);
- **S-factor** ← `subject ⇒ label` (SCPS) and other `Y=g(D)` degeneracies;
- **S-support** ← unseen device/montage regimes with target feature mass off the source manifold.

Each mechanism must be routed through §3 before any adaptation claim: name the shift object,
the regime, the contracts, the identifiable estimand, and the certificate that fires on failure.
