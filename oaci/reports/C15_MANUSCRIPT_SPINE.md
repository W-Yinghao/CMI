# C15 — Manuscript spine

**Working title:** *When Source-Side Invariance Does Not Transfer: A Falsification Battery for EEG Domain
Generalization under Support Mismatch.*

**One-sentence contribution.** We do not propose another DG penalty; we propose a **falsification battery** —
support-aware leakage measurement, selector-oracle replay, and source→target anti-transfer diagnostics — that
makes the failure of source-side DG *control* mechanisms **localizable, reproducible, and reusable** under
domain–class support mismatch, and we apply it to falsify two such mechanisms on BNCI2014-001.

**Canonical framing sentence (use verbatim at abstract level):**
> These experiments do not show that support-aware invariance is useless; they show that, under BNCI2014-001
> LOSO with strict target isolation, the tested source-side control mechanisms fail to transfer to target
> worst-domain endpoints. The contribution is a falsification battery that makes this failure localizable,
> reproducible, and reusable.

## Claims (reviewer-hardened; each grounded in a committed report — see C15_CLAIM_EVIDENCE_MAP.json)

> These claims were **adversarially stress-tested** (independent skeptics tried to refute each on the
> committed numbers). All 4 survive at *minor over-claim risk* **once scoped as below**. Every headline is
> scoped to **BNCI2014-001 LOSO / ShallowConvNet** (C3 additionally to **seed 0, n=6**). See
> `C15_REVIEWER_OBJECTION_MATRIX.md` for the full objection table and the genuine evidence gaps.

**Claim 1 — Measurement (scoped).** On BNCI2014-001 LOSO (ShallowConvNet, seeds 0–2, one fixed probe family),
a support-aware, source-only leakage diagnostic yields **reproducible, source-isolated, auditable**
measurements — a **null** held-out audit signal (**C8: 0/54 BH-FDR survivors**) alongside a large measured
selection-time reduction (**C10a: Δsel −0.326, 54/54**). *We do NOT demonstrate that a naive estimator fails
or quantify support mismatch on this (balanced 4-class) dataset; the measurement is descriptive and
probe-relative.*

**Claim 2 — Control-failure localization (scoped).** On BNCI2014-001 LOSO, the battery *localizes* OACI's
control failure: selection reductions do not survive audit (**Δsel −0.326 (54/54) vs Δaudit +0.008 (25/54),
corr +0.004**); nominal audit signals do not survive multiplicity (**11 nominal, 0 Bonferroni, 0 BH / 54**);
the pre-registered K2 returns **stop_no_reproducible_gain** (worst-domain bAcc 2/6 improved; worst-domain NLL
improves on average **but not reproducibly**, worst-fold +0.32); and a **diagnostic, non-deployable
source-audit oracle** cannot identify a gain-reproducing checkpoint **from held-out source signal** (**C10b
`C_oracle_also_fails`**, replay 216/216, 0 flips, S0-replay == C8 K2). *This is NOT evidence that DG, EEG
transfer, or a target-informed selector fails; no target-side oracle was run.*

**Claim 3 — Anti-transfer (scoped, seed 0).** On BNCI2014-001 LOSO (ShallowConvNet, **seed 0**), across all
6 actively-trained cells (3 targets × 2 τ_lse), SRC drove source-guard NLL down ~1 nat (to ~0.09, **guard
memorization**) while worsening target worst-domain NLL in **every** cell — source→target **anti-transfer**
(**ATI 1.0, STI 1.0, source_nll→target_nll pearson −0.947, spearman −1.0, n=6**). *Single seed, n=6, no CI;
only τ swept — no λ/lr/regularization sweep, so anti-transfer under a well-regularized SRC is untested; a
stress replication, not a law. (4/6 blowups are SRC-caused; 2 are ERM-fallback cells already above uniform.)*

**Claim 4 — Framework (scoped).** The reusable contribution is a falsification battery (G0–G5), **instantiated
once** (BNCI2014-001 LOSO, ShallowConvNet) to **falsify** two source-side control mechanisms (OACI, SRC)
**under this protocol**. *Evidence:* C14 verdict = **falsified** [no_endpoint_transfer, oracle_failure,
source_target_antitransfer]; method-closure forbids an OACI-v2 selector; support-aware leakage
retained_as_measurement. *We do NOT claim the battery is validated across datasets/backbones or that any DG
penalty must fail; it has only ever returned "falsified" (no positive control) — discriminative validity is
future work.*

## Outline

1. **Introduction.** DG penalties are justified by source-side invariance / robustness. Under domain–class
   support mismatch these control signals may be non-identifiable, overfit, or anti-transfer. We propose a
   falsification battery rather than another penalty.
2. **Problem: support mismatch and conditional leakage.** domain×class support graph; estimable vs unsupported
   cells; why smoothing unsupported cells is a scientific error; the source-only audit requirement.
3. **Falsification battery.** G0 integrity/target isolation · G1 selection→audit optimism · G2 held-out
   leakage (K1, grouped permutation + multiplicity) · G3 endpoint transfer (K2, worst-domain) · G4
   source-audit oracle replay · G5 source→target anti-transfer.
4. **Experimental setting.** BNCI2014-001 LOSO; source_train / source_audit / target_audit isolation; ERM,
   OACI, oracle selectors, SRC; deterministic artifacts + byte/numeric replay identity.
5. **Results.** C8 (weak nominal audit leakage, no multiplicity-stable K1, no K2 gain); C10 (oracle fails to
   rescue); C12 (source endpoint improvement → target anti-transfer); C14 (method closure + falsification
   verdicts).
6. **Discussion.** What is falsified: the *tested source-side control mechanisms*. What is not falsified: all
   DG, all EEG transfer, the support-aware diagnostics. Why measurement-first DG is valuable.
7. **Limitations.** one dataset family; ShallowConvNet / current protocol; no claim that every DG penalty
   fails; the oracle is diagnostic, not deployable. (See C15_tables/limitation_boundary_table.csv.)
8. **Conclusion.** Support-aware falsification should precede claims of DG *control* under support mismatch.

## Evidence chain (see c15_tables/evidence_chain_c8_c10_c12_c14.csv)
C8 → C10a → C10b → C12 → C14, each committed with report + tests; every number in the claim map is pulled
from a committed report by `oaci/confirmatory/c15_spine.py` (no transcription).

## Say / do-not-say

**Say:** "OACI and SRC are closed as control objectives under this protocol, but support-aware leakage,
oracle replay, and anti-transfer diagnostics are retained as falsification instruments."

**Do NOT say:** ~~OACI failed, so DG is bad.~~ ~~EEG DG does not work.~~ ~~Invariance is useless.~~ ~~OACI is
mathematically wrong.~~

## Genuine evidence gaps (future work — NOT current claims; from the adversarial pass)
These are honestly outside the committed evidence and must be presented as future work, not assertions:
- **Support-mismatch existence** on BNCI2014-001 is unquantified (no committed n_unsupported/n_estimable; it
  is balanced 4-class MI) — the distinctive machinery may not be *exercised* here.
- **No naive-vs-support-aware / ungrouped-vs-grouped contrast** — "ill-posed" is a motivating premise.
- **Probe-relativity** of `L_Q^ov` unquantified (probe family + reference prior fixed by protocol).
- **No target-side oracle** — case C is w.r.t. held-out *source* signal only.
- **SRC anti-transfer is seed-0, n=6, no CI, no λ/lr/regularization sweep** — source guard NLL collapses to
  ~0.09 (memorization); anti-transfer under a well-regularized SRC is the deepest open question.
- **No positive control / discriminative validity** — the battery has only ever returned "falsified"; no
  ERM-beating method has been run through it to show the gates certify transfer, not just flag failure.
- **Single dataset family + backbone + minimum seeds** — BNCI2014-004 and seeds [3,4] are barred under the
  pre-registered pause; a second dataset with genuine support mismatch is the key generality test.

## Scope note
This is a manuscript SPINE (structure + claims + evidence map), not a draft. No new experiments were run to
produce it. New hypotheses (TTA-without-leakage, abstention/calibration, hierarchical pooling) are deferred
until this falsification framework is packaged as the primary contribution.
