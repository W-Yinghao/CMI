# Project A — Experimental Protocol

> Experiments in Project A are **observability audits**, not just performance comparisons: each
> reports *observed information + contracts + estimator → allowed claim*. This protocol fixes the
> per-experiment claim ledger, the three experiment tiers, the required experiments, and the
> hard reporting rules that forbid reporting R0/R1 metrics as target guarantees. Uses
> `06_oaci_identifiability.md` (`OA-0`, checkable-vs-assumed) and `05_csc_shift_calculus.md`.
> Naming per `00_repo_audit.md §5`. No training-code changes are implied by this file.

## 0. Purpose

Every experiment answers: *given the observed regime `R`, the declared contracts `C`, and the
estimator, which claims are admissible under `OA-0`?* Performance numbers are inputs to that
question, not the answer.

## 1. Required claim ledger

Every experiment must report (machine-checkable fields in §5):
```
Regime:                 R0 | R1 | R2
Observed coordinates:   (which parts of O_R are actually used)
Contracts invoked:      (subset of C1..C12)
Checkable contracts:    (invoked ∩ checkable in R)
Uncheckable contracts:  (invoked \ checkable in R  — assumed, not evidenced)
Identifiable estimand:  (the T that (R,C) pins down under OA-0; or an identified set)
Estimator:              (f / critic; C5 fidelity diagnostics)
Failure certificates:   (which CE/P0 fires if a contract breaks)
Forbidden claims:       (the overclaims this design must NOT make — 05 §6)
```

## 2. Experiment tiers

- **Tier 0 — exact discrete certificates.** `counterexamples/run_counterexamples.py`; **no
  training**; validates the non-identifiability *arithmetic*. These are proofs.
- **Tier 1 — `h2cmi` simulator illustrations.** `h2cmi.data.eeg_simulator`; vary
  `prior/concept/montage/noise/label_mechanism`. **Oracle labels (`ystar`) may be used only for
  *validation*, never as R1 evidence.** Illustrations, not proofs.
- **Tier 2 — real EEG audit.** source-only LOSO → **source claim only**; target-unlabeled →
  **marginal / support / prior-under-contract only**; minimal-paired → **paired transport /
  bounded risk only if anchors satisfy C8 ∧ C11**.

## 3. Required experiments

- **E0 — exact certificates.** CE-R0-1/2/3, CE-R1-1/2, CE-C1-1, CE-MP-1, CE-C11-1, CE-MONO-1
  (all in Tier 0; all currently PASS).
- **E1 — `TOS-1` source-only audit.** Show source LOSO metrics **cannot** be reported as
  target-specific gain (regime R0; forbidden claim = target gain). Certificate: `TOS-1` / CE-R0-2.
- **E2 — R1 prior-identifiability stress.** Vary the rank of the mixture matrix `B` and the
  support overlap; show `π_T` recovery **only** when C1 ∧ C2 ∧ C3 hold (`TU-1`), and
  non-recovery at rank-deficiency (CE-R1-2) or support failure (CE-C1-1).
- **E3 — R1 concept non-identifiability.** Same target `X`/`Z`, different target labels; show any
  R1-only procedure sees the **same** observation (`TU-2` / CE-R1-1); forbidden claim = "concept
  detected from unlabeled target".
- **E4 — R2 transport-rank study.** Vary anchor count `k` and dimension `p`. If `k<p` (or
  anchors rank-deficient / invalid), the transform is non-identifiable (CE-MP-1 / CE-C11-1); if
  the anchor matrix is full-rank and the transform family low-dimensional, the transform is
  identifiable up to estimation error (`MP-1`).
- **E5 — safety-gate falsification.** Use CE-R0-2 and simulator analogues to show the source
  gain sign does **not** identify the target gain sign (C9 / measurement→control gap).

## 4. Reporting rules (hard)

**Balanced accuracy** may be reported as:
- a **source** validation metric under R0;
- an **oracle** simulator check, *explicitly marked oracle* (Tier 1);
- a **target-label** metric only under R2 / evaluation-only held-out labels.

> It must **never** be reported as identifiable from R1 unlabeled target data, nor may an R0
> source metric be reported as a target risk / gain / concept guarantee.

**Leakage metrics** (`I(Z;D|Y)`, `I(Y;D|Z)`) may be reported as **diagnostics only**, never as
risk or accuracy guarantees (CSC-R2; C5 governs whether the measured value is even the population
value). `I(Y;D|Z)` is reported as a predictive-insufficiency residual, never as "concept detected"
outside R2 + C4 ∧ C6 ∧ C5.

## 5. Minimal output format

Each run should save an observability report (`observability_report.json` + `.md`) with fields:
```
regime
contracts_invoked
checkable_contracts
uncheckable_contracts
identifiable_estimand          # or identified_set
observation_used
estimator                      # incl. C5 fidelity diagnostics (stepA_dom_acc, probe gap)
certificate_passed             # which CE/P0 checks ran and their result
forbidden_claims_checked       # the 05 §6 list, each asserted NOT made
oracle_fields_used_for_validation_only   # e.g. ystar in Tier 1 — flagged, never as evidence
```
The report makes the *claim boundary* auditable independently of the accuracy table: a reviewer
can verify that every asserted estimand is licensed by `(regime, contracts)` under `OA-0`, and
that every forbidden claim was explicitly checked and not made.

---

**Scope.** This protocol governs how results are *reported and bounded*; it does not itself run
experiments or modify training code. Tier 0 is live (`run_counterexamples.py`); Tiers 1–2 are the
template for any future simulator / real-EEG audit under Project A.
