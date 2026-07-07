# FSR_33 — Final Story Lock (Phase 6A)

**Project FSR.** The manuscript story is **LOCKED** (PM decision). All prose in `paper/fsr/` must conform to this.

## Title (locked)
> **When Subject Leakage Becomes a Shortcut: Verification, Refusal, and Scoped Repair in EEG Representations**

## One-sentence thesis
EEG subject leakage is neither automatically harmful nor automatically benign; FSR is a **functional shortcut
verification framework** that decides *when* to refuse a blind repair and *when* a repair holds only within a
narrow mechanistic scope.

## The four results (locked order + evidence)
1. **Measurable ≠ reliance.** Raw leakage magnitude does not certify functional reliance; task-head alignment is
   the correctly-signed predictor but not a validated estimator. *(CIGL / R3; C1–C2.)*
2. **Erasable ≠ beneficial.** Subject signal is erasable, but no eraser achieves a proven target benefit
   (`benefit_claimable = 0/40`); erasure strength does not certify a target gain. *(TOS; C3–C5.)*
3. **Natural branch-local verification refuses blind repair.** The spatial branch is the strongest subject-
   leakage and load-bearing candidate, but subject-subspace removal *hurts* the target, so FSR refuses to call it
   a harmful shortcut — it is task-entangled. *(Phase 4B refit; C6, C16; supersedes the old RQ4-blocked C7.)*
4. **Repair has a scoped boundary.** A controlled harmful shortcut can be detected, localized, and exactly
   attributed (PC1); a controlled **first-moment deterministic** offset is repairable by target-X-only mean
   alignment (4F, construction-matched, narrow); a controlled **second-moment stochastic** perturbation is
   **not** repaired by covariance-shrinkage at the operating point, even oracle-directed (4G). *(PC1/4D/4E/4F/4G;
   C11–C14, C17.)*

## Locked takeaway
> FSR does not merely say "do not repair." It says **when to refuse a blind repair** (natural spatial leakage is
> task-entangled), and **when a repair holds only within a mechanistic scope** (first-moment deterministic
> offsets, not second-moment stochastic perturbations, and not natural/learned shortcuts). The
> measurement→control gap is pushed to the intervention layer and mechanistically bounded.

## Framing discipline (isomorphic to Prior-Decoupled TTA)
A joint measurement is not an action command: it must be decomposed into its effect pathway. FSR decomposes a
shortcut into leakage (L1), reducibility (L2), erasability (L3), task coupling (L4), functional reliance (L5),
target consequence (L6), and — new — **repair scope**. Contribution = the audit ladder + the two boundary
findings (*measurable ≠ relied-upon*, *erasable ≠ beneficial*) + the natural-verification refusal + the scoped
repair boundary. **Not** a new DG method, **not** SOTA, **not** a CMI-control revival.

## Forbidden ↔ allowed sentences (binding; see FSR_35 for the reviewer risks)
Forbidden: "FSR repairs EEG shortcuts" · "E4 repairs natural subject leakage" · "E4b repairs second-moment
shortcuts" · "FSR is a new EEG-DG method" · "CMI improves EEG generalization" · "spatial/subject leakage is
harmful" · "erasure is a repair."
Allowed: "FSR verifies shortcut claims before repair" · "natural spatial subject leakage is task-entangled and
not verified harmful" · "E4 repairs a controlled first-moment constant-offset shortcut with narrow scope" · "E4b
does not repair a controlled second-moment stochastic shortcut at the source-selected operating point" · "repair
beyond first-moment deterministic offsets remains open."
