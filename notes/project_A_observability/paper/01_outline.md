# Proposed Manuscript Outline

## Title

**EEG Adaptation Observability and Information Contracts**

## Abstract skeleton

- **Problem.** EEG adaptation (cross-subject / cross-session domain generalization and test-time
  adaptation) is routinely *evaluated* with target labels but *deployed* under source-only or
  target-unlabeled information.
- **Gap.** Evaluation metrics are often reported as if they were identifiable under the deployment
  regime; there is no shared vocabulary for what a given (regime, contract) actually licenses.
- **Method.** Observability-Aware Contracted Identifiability (OACI) + a contracted shift calculus +
  prior-decoupled CMI, made executable as machine-checked observability reports.
- **Results.** Exact indistinguishable-world counterexamples for each non-identifiability claim; a
  machine-checked claim ledger; audited real MOABB motor-imagery mini-grids with a chance-normalized
  multi-dataset digest.
- **Conclusion.** What can be claimed is fixed by observed information and declared contracts, not by
  the evaluation harness. Target metrics computed with oracle labels are reportable but not
  identifiable under source-only / target-unlabeled deployment.

## Sections

### 1. Introduction
- The evaluation-vs-deployment information gap in EEG adaptation. Draws: `00_repo_audit.md`, `00_claims_and_contributions.md`.
- Reportable vs identifiable; a one-paragraph statement of the thesis.
- Contributions C1–C6.
- Explicit non-claims (no SOTA, no source-only target-gain identification).

### 2. Background and failure modes
- CMI leakage `I(Z;D|Y)`, hierarchical-D, TTA + safety gate as the substrate (`h2cmi`).
- The retracted claims that motivate the project: posterior-KL upper bound, zero-Bayes-error escape,
  concept-shift-from-`I(Y;D|Z)`, source-only target prior. Draws: `h2cmi/THEORY.md` P0-2..P0-5,
  `04_prior_decoupled_theory.md §1`.

### 3. Observability regimes
- World model; `Z=f(X)` is chosen not primitive. Draws: `01_information_regimes.md §1`.
- R0 source-only / R1 target-unlabeled / R2 minimal-paired; observation operators `O_R`.
- MONO-1 information monotonicity + the non-interchangeability corollary (source breadth ≠ target
  observation, CE-MONO-1). Draws: `01 §5–6`, `06 §8`.

### 4. OACI identifiability
- World class `M(C)`, compatibility set `K_{R,C}`, OA-0 definition, the certificate pattern.
- Partial identification (identified set `I_{R,C}`), contract-strength monotonicity, target-law-axiom
  caveat. Draws: `06_oaci_identifiability.md`.

### 5. Contracted shift calculus (CSC)
- Factorized world; shift objects (acquisition/prior/concept/support/transport/rater/factor) → regime
  → contracts → estimand → certificate. Allowed vs forbidden diagnostic vocabulary.
  Draws: `05_csc_shift_calculus.md`.

### 6. Prior-decoupled CMI
- ID-1 identity; reweighting `w_d(y)=π*(y)/π_d(y)` ⇒ `Ĩ(Y;D)=0`; PD-1 additive relation.
- What PD-1 does NOT say (no accuracy, no concept shift, source-side only). TU-1 target-prior
  identifiability under R1. Draws: `04_prior_decoupled_theory.md`.

### 7. Executable audit layer
- Claim → Verdict → ObservabilityReport; the rule engine (R0 target estimands rejected; TU-1 for
  target prior; leakage diagnostic; oracle/eval-only markings). Forbidden-claim list + validator +
  tracked digest. Draws: `h2cmi/observability/`, `08_experimental_protocol.md §6`.

### 8. Experiments
- Tier 0 exact certificates (`run_counterexamples.py`); tier 1 simulator illustrations.
- Real MOABB audited grids: Step 8/9 (`BNCI2014_001`, 4-class), Step 10 (`BNCI2014_004`, binary;
  `BNCI2015_001` legal-skip). Chance-normalized multi-dataset digest. Draws: `05_experiment_table.md`,
  `06_results_digest.md`.

### 9. Limitations
- MOABB-motor-imagery-only; modest (non-SOTA-tuned) training; oracle labels evaluation-only;
  counterexamples are the proof layer. Draws: `07_limitations_and_claim_boundary.md`.

### 10. Conclusion
- Observed information + declared contracts determine what can be claimed; a reusable audit contract
  for adaptation papers.

## Section → source-file map (quick reference)

| Section | Primary source files |
|---|---|
| 1 | `00_claims_and_contributions.md`, `00_repo_audit.md` |
| 2 | `h2cmi/THEORY.md`, `04 §1` |
| 3 | `01_information_regimes.md` |
| 4 | `06_oaci_identifiability.md` |
| 5 | `05_csc_shift_calculus.md`, `02_contract_taxonomy.md` |
| 6 | `04_prior_decoupled_theory.md` |
| 7 | `h2cmi/observability/`, `08_experimental_protocol.md` |
| 8 | `07_counterexample_catalog.md`, `05_experiment_table.md`, `06_results_digest.md` |
| 9 | `07_limitations_and_claim_boundary.md` |
