# When Source-Side Invariance Does Not Transfer: A Falsification Battery for EEG Domain Generalization with Support-Aware Leakage Diagnostics

> **Manuscript draft v0.1 (C16).** Assembled from the C16 section files; every number is grounded in the committed C8/C10/C12/C14/C15 reports (none transcribed by hand). No new experiments were run. Drafting order was Results → Methods → Limitations → Introduction → Abstract (to keep each sentence anchored to evidence); presented here in manuscript order. Claim-language safety rail: `C16_CLAIM_LANGUAGE_AUDIT.md`. Reviewer-objection matrix: `../reports/C15_REVIEWER_OBJECTION_MATRIX.md`.


## Abstract

Source-side invariance and robustness penalties are widely used in domain generalization (DG), but their
control signals — reduced conditional-domain leakage, improved source robustness, or better held-out source
endpoints — are often trusted before they are falsified against held-out *target* endpoints. We present a
falsification battery for EEG domain generalization that audits source-side control mechanisms under strict
target isolation. The battery combines support-aware conditional-leakage measurement (restricted to estimable
domain–class cells, with a grouped permutation null), a selection→audit optimism check, a
multiplicity-controlled held-out leakage test, a reproducible worst-domain endpoint test, a non-deployable
source-audit oracle replay, and a source→target anti-transfer diagnostic. Applied to BNCI2014-001
leave-one-subject-out motor imagery with a ShallowConvNet backbone, the battery localizes the failure of two
source-side control hypotheses. A conditional-domain leakage-control method reduces leakage at selection time
on every fold (Δ = −0.326, 54/54) but yields no multiplicity-stable held-out leakage evidence (0 of 54
Benjamini–Hochberg survivors) and no reproducible target worst-domain endpoint gain, and a non-deployable
source-audit oracle — replayed identity-exact over the method's own trajectory — does not rescue it. A
separate source-robust endpoint objective improves the source worst-domain endpoint (by ≈ 1 nat) under the
tested configuration while worsening the target worst-domain endpoint in every active cell, exposing
source→target anti-transfer. These results do not show that domain generalization, EEG transfer, or
support-aware invariance is impossible; they show that, under this protocol, the tested source-side control
signals should be retained as **falsification and measurement instruments** rather than trusted as deployable
control objectives. We release the battery and the fully identity-checked evidence chain so that a proposed DG
penalty can be audited before it is believed.


## Introduction

Domain-generalization (DG) methods for EEG and other physiological signals are frequently justified by a
**source-side control signal**: a quantity computable from the source domains alone — reduced
conditional-domain leakage (invariance), improved source robustness, or a better held-out source endpoint —
that is expected to translate into better performance on an unseen target subject or site. Such signals are
appealing precisely because they need no target labels. That same property makes them easy to *trust before
they are tested*: a penalty that improves a source-side signal, and a model selected to optimize it, can look
successful without any evidence that the improvement reaches the target.

Under strict target isolation, these signals can be **audited before they are trusted**. Two obstacles make a
careful audit necessary. First, under **domain–class support mismatch** — some `(domain, class)` cells having
little or no source support — conditional-invariance diagnostics that average over all cells are not
well-identified, so leakage must be measured only on estimable cells with an honest permutation null. Second,
a single point estimate on a selection split conflates a real effect with selection-time optimism, and a
method's failure to beat a baseline can always be blamed on a poor selector or a poor selection split.

We propose a **falsification battery** that tests whether a source-side control signal transfers to
**target worst-domain endpoints**, and that closes the usual escape hatches. The battery combines: a
support-aware, source-only extractable-leakage measurement; a selection→audit optimism check; a
multiplicity-controlled held-out leakage test (K1); a reproducible worst-domain endpoint test (K2); a
**source-audit oracle replay** that removes selector and selection-split quality as explanations; and a
**source→target anti-transfer** diagnostic for source-endpoint objectives. Every gate reads only source
information for its decisions; the target is used strictly for evaluation.

We instantiate the battery once, on BNCI2014-001 leave-one-subject-out motor imagery with a ShallowConvNet
backbone, and use it to falsify two plausible source-side control mechanisms as case studies: a
conditional-domain leakage-minimization method (OACI) and a source-robust endpoint objective (SRC). The
battery localizes the failure — selection leakage reductions that do not survive audit, held-out signals that
do not survive multiplicity, no reproducible endpoint gain, an oracle that cannot rescue the trajectory, and a
source endpoint objective that *anti-transfers* to the target. The point of the paper is not that these two
methods lose to ERM; it is that a support-aware, target-isolated falsification protocol makes *why* they
should not be trusted **localizable, reproducible, and reusable**.

**Contributions.**
1. A falsification battery (G0–G5) for source-side DG control signals under strict target isolation, built on
   support-aware conditional-leakage measurement, held-out leakage/endpoint gates, source-audit oracle replay,
   and source→target anti-transfer diagnostics.
2. Two case-study falsifications on BNCI2014-001 LOSO — OACI (leakage-control) and SRC (endpoint-control) —
   each localized to a specific gate, with byte/numeric-exact replay identity and pre-registered decisions.
3. A stance and a set of retained instruments: support-aware leakage and the K1/K2/oracle/anti-transfer
   diagnostics are kept as **measurement and falsification** tools, distinct from the control objectives they
   falsify. We are explicit about scope: these are single-dataset, single-backbone results; the battery has so
   far only returned "falsified," and its discriminative validity is future work.

We deliberately avoid the over-generalizations that the evidence does not support: we do not claim that DG or
EEG transfer is impossible, that support-aware invariance is useless, or that BNCI2014-001 by itself
establishes a support-mismatch regime.


## Problem setting and the falsification battery

## Problem setting: support mismatch and conditional leakage

Many domain-generalization (DG) penalties are justified by a **source-side control signal**: reduced
conditional-domain "leakage" (invariance), improved source robustness, or better held-out source endpoints.
The premise is that improving such a signal on the source domains will improve worst-domain performance on an
unseen target domain. We take that premise as a *hypothesis to be falsified*, not assumed.

A conditional-invariance objective aims to make a representation `Z` uninformative about the domain `D` given
the label `Y`. Under **domain–class support mismatch** — where some `(domain, class)` cells have little or no
source support — the population quantity `p(z | y, d)` is not defined for unsupported cells, so a diagnostic
that averages a per-cell alignment term over *all* cells (implicitly smoothing over cells with no support) is
not well-identified. We therefore restrict measurement to **estimable cells** (a support graph separates
estimable from unsupported cells; unsupported cells are flagged, never smoothed), and we measure a
**probe-extractable** quantity: `L_Q^ov`, the extractable conditional-domain information recovered by a
grouped cross-fit domain probe on frozen `Z`, restricted to estimable cells and evaluated against a
recording-grouped permutation null. Crucially, the audit is performed on a **held-out source split**
(`source_audit`); the target is never read.

*Scope note (carried from the adversarial review).* On the balanced 4-class BNCI2014-001 data used here, we do
not separately quantify how many cells are unsupported; "ill-posed under support mismatch" is the *motivation*
for the support-aware construction, and BNCI2014-001 alone does not establish a support-mismatch regime. The
support-aware diagnostic is what we measure with; its distinctive value under genuine mismatch is future work.

## The falsification battery

The battery is a fixed sequence of six gates. Each gate is a pure function of committed, deep-verified
artifacts; the battery reads no target information for any selection decision.

- **G0 — Integrity.** Are the artifacts deep-verified, target-isolated (`target_fit_ids = ∅`), and is the
  checkpoint replay identity-exact? A failure here means the downstream gates would run on untrustworthy
  evidence. *(Outcome: `integrity_ok`.)*
- **G1 — Selection→audit optimism.** Does a selection-time leakage reduction survive at the held-out source
  audit? We compare the per-fold change in selection leakage against the change in audit leakage.
- **G2 — Held-out leakage (K1).** Does the held-out audit leakage reduction survive multiplicity? K1 is a
  recording-grouped 2000-permutation null per fold; the sweep-level line applies Bonferroni and
  Benjamini–Hochberg control. A *weak nominal, non-multiplicity-stable* signal is **not** a success.
- **G3 — Endpoint transfer (K2).** Does any leakage signal convert to a reproducible worst-domain endpoint
  gain over ERM, on both `worst_domain_bacc` and `worst_domain_nll`, at every (seed, level) unit?
- **G4 — Source-audit oracle replay.** Replaying selectors over the method's own risk-feasible trajectory, can
  a **non-deployable source-audit oracle** — which may read the held-out `source_audit` split but never the
  target — identify a gain-reproducing checkpoint? This removes "bad selector" and "bad selection split" as
  explanations. The oracle is a diagnostic upper bound on *source-identifiable* rescue, not a target oracle.
- **G5 — Source→target transfer.** For a source-side *endpoint* objective, does improving the source
  worst-domain endpoint transfer to the target, stay flat, or **anti-transfer** (source improves, target
  worsens)? We report an anti-transfer index (fraction of active cells with source-improvement and
  target-harm) and a source→target instability score.

The battery verdict is `control_hypothesis_supported` only if the endpoint gate certifies a reproducible gain
*and* the oracle can rescue; otherwise it lists the specific falsification reasons
(`falsified_by_no_endpoint_transfer`, `falsified_by_oracle_failure`, `falsified_by_source_target_antitransfer`,
`falsified_by_selection_optimism`). We emphasize that the battery has, to date, only ever returned
`falsified`; a **positive control** — an ERM-beating method certified by the same gates — is not yet available,
so the battery's *discriminative* validity (its ability to certify a genuinely transferring method rather than
only flag failures) remains future work.

## Experimental protocol

- **Data / task.** BNCI2014-001, 9 subjects, 4-class motor imagery, LOSO (each subject held out in turn).
- **Roles.** `source_train` / `source_guard` / `source_audit` / `target_audit`, with `target_audit` used for
  evaluation only. Provenance tracking asserts `target_fit_ids = ∅` (the target never enters any fit).
- **Backbone.** ShallowConvNet; a shared Stage-1 ERM checkpoint per fold, with Stage-2 method objectives
  starting from it under a risk-feasibility constraint `R_src ≤ R_ERM + ε`.
- **Methods.** ERM (baseline), OACI (conditional-domain leakage control), `global_lpc` / `uniform` (posterior
  / uniform alignment baselines), and SRC (source-robust endpoint control). OACI/`global_lpc`/`uniform` were
  evaluated in the confirmatory K1/K2 run (seeds [0,1,2]); SRC in a dedicated one-fold pilot and a 3-target ×
  2-temperature stress replication (seed 0).
- **Reproducibility.** Every fold artifact is deep-verified; the counterfactual selector replay reproduces the
  stored per-checkpoint predictions with **0 argmax flips** (byte-hash where the GPU arch matches the original
  node, numeric to ~10⁻¹⁵ otherwise). All decision numbers are pulled from committed artifacts, not
  transcribed.
- **Pre-registration and scope.** K1/K2 thresholds and the SRC selector guards are fixed in a manifest. Under
  a pre-registered pause, we do **not** run additional seeds or a second dataset (BNCI2014-004) as part of this
  work; the current claims are scoped to BNCI2014-001 / ShallowConvNet accordingly.


## Results

All experiments use BNCI2014-001 (9-subject, 4-class motor imagery) in a leave-one-subject-out (LOSO)
protocol with a ShallowConvNet backbone, under strict role isolation: `source_train` (training),
`source_guard` (per-level source held-out for endpoint guards), `source_audit` (source held-out for the
leakage audit), and `target_audit` (the held-out subject, used for **evaluation only** and never read during
training or selection). Every artifact is deep-verified and every replayed prediction is identity-checked
(Section: Experimental protocol). We report four results; each closes one link in the chain from a source-side
control signal to a target-side benefit.

## Result 1 — Selection-time leakage reduction does not become stable held-out audit evidence

OACI reduces the support-aware extractable leakage `L_Q^ov` at *selection* time on every fold: across the 54
fold-levels (9 targets × 2 deletion levels × 3 seeds, summarized as 54 selection comparisons), the mean
change in selection leakage relative to ERM is **Δ = −0.326**, reduced in **54/54** cases. On the *held-out
source-audit* split, however, the same mean change is **+0.008**, reduced in only **25/54** cases, and the
per-fold selection and audit changes are essentially uncorrelated (**Pearson r = +0.004**). The pre-registered
K1 test — a recording-grouped, 2000-permutation null on `source_audit` only — yields **11 nominal detections
out of 54 one-sided tests**, but **0 survive Bonferroni and 0 survive Benjamini–Hochberg** control; the
sweep-level verdict is `stop_no_detectable_heldout_leakage_reduction`.

> **Interpretation.** This does *not* show that support-aware leakage is meaningless or that OACI cannot reduce
> leakage. It shows that selection-time leakage reduction, taken alone, is **insufficient evidence** for a
> transferable DG control mechanism: the reduction is a selection-time optimism that does not reproduce as a
> multiplicity-stable held-out audit signal. A null held-out audit result is itself a valid, source-isolated,
> auditable measurement.

## Result 2 — Held-out leakage does not convert to reproducible target endpoint gains

The pre-registered K2 endpoint test asks whether any leakage signal converts to a reproducible worst-domain
gain over ERM, on both endpoints (`worst_domain_bacc`, higher better; `worst_domain_nll`, lower better),
requiring a gain at **every** (seed, level) unit with `min_seeds = 3`. The verdict is
**`stop_no_reproducible_gain`**. On the primary accuracy endpoint, worst-domain balanced accuracy improves in
**2 of 6** units and is harmed in **4 of 6** (mean Δ = **−0.005**). On the calibration endpoint, worst-domain
NLL improves on average (mean Δ ≈ **−0.107**, improving in 4 of 6 units) **but is not reproducible**: two
units are harmed, with a worst-fold degradation of **+0.32**, so the both-levels criterion is not met.

> **Interpretation.** The K2 test closes the downstream-benefit claim: target worst-domain endpoints do not
> improve *reproducibly*. We report the calibration endpoint honestly — some worst-domain NLL comparisons move
> in the favorable direction — but the aggregate, pre-registered endpoint criterion does not support a
> reproducible target gain, and we do not cherry-pick the favorable endpoint.

## Result 3 — A source-audit oracle cannot rescue the OACI trajectory

To remove the "the selector was simply bad" escape hatch, we replay six checkpoint selectors over OACI's own
risk-feasible trajectory, evaluating each choice on the target. The replay is byte/numeric identity-exact
against the stored artifacts (**216/216 selected-checkpoint checks pass, 0 argmax flips, max |Δlogit| ≈
1.8×10⁻¹⁵**), and the current-selection replay (S0) reproduces the C8 K2 verdict exactly — an internal
consistency check. Four source-only guard selectors (S1–S4) and, critically, a **non-deployable source-audit
oracle (S5)** — which is allowed to read the held-out `source_audit` split but never the target — all return
`stop_no_reproducible_gain`. The oracle selects an OACI (non-ERM) trajectory checkpoint in 51 of 54
fold-levels, yet still produces no reproducible target gain.

> **Interpretation.** The oracle is **not a target oracle**; it tests whether *stronger held-out source
> information* could have rescued the source-side control hypothesis. It cannot. We therefore state the finding
> precisely: **no checkpoint in the trajectory is rescuable by the tested held-out source-audit oracle** — not
> that no rescuing checkpoint exists in any sense. This removes selector quality and selection-split quality as
> explanations of the failure, under this protocol.

## Result 4 — A source-endpoint objective anti-transfers to the target

We then test a *different* source-side control signal: SRC, a source-robust objective that directly minimizes
a smooth worst-domain balanced cross-entropy over source domains under the same risk-feasibility constraint,
selected by a source-train-only endpoint guard. Across all **6 actively-trained cells** (3 targets × 2
smoothing temperatures, seed 0), SRC **improves the source worst-domain endpoint** — driving source-guard NLL
down by roughly **1 nat**, to ≈ 0.09, a value consistent with near-memorization of the source guard — **while
worsening the target worst-domain NLL in every one of the 6 cells**. The source→target NLL relationship is
strongly negative (Pearson **r = −0.947**, Spearman **−1.0**, n = 6; anti-transfer index and instability score
both **1.0** over these 6 cells). SRC-caused target-NLL blowups (target NLL above the uniform-prediction bound)
occur in **4 of 6** active cells; the gentler smoothing temperature does not avoid them, and at the missing-cell
level SRC's guard admits no candidate, so it falls back to ERM.

> **Interpretation.** This is evidence of **anti-transfer under the tested SRC configuration** — optimizing a
> source-side endpoint made the target endpoint *worse* — not a proof that all well-regularized source-robust
> objectives anti-transfer. The source-guard collapse to ≈ 0.09 indicates guard memorization; because only the
> smoothing temperature (not weight-decay / learning-rate / other regularization) was varied, and the result
> rests on a single seed with n = 6, we scope the claim to the tested configuration.

## Summary

Under this protocol, each link from a source-side control signal to a target benefit is broken: selection
leakage reductions do not survive audit (R1); nominal audit signals do not convert to reproducible endpoint
gains (R2); a source-audit oracle cannot rescue the trajectory (R3); and a distinct source-endpoint objective
anti-transfers (R4). The falsification battery (Methods) records these as gate outcomes G1–G5 and returns the
verdict `control_hypothesis_falsified` — while the support-aware leakage measurement, the K1/K2 gates, the
oracle replay, and the anti-transfer diagnostic are all retained as instruments.


## Discussion

Our results are best read not as "OACI underperformed ERM" but as a **localization of where a source-side
control hypothesis breaks** under a domain-generalization protocol with strict target isolation. The battery
converts a negative outcome into a set of specific, reproducible falsifications, and — equally important — into
a set of instruments that are *retained* precisely because they made the failure visible.

**What is closed (under this protocol).**
- OACI as a *leakage-control* method for downstream target benefit: selection-time leakage reductions do not
  survive audit (G1), audit signals are not multiplicity-stable (G2), and they do not convert to reproducible
  target worst-domain gains (G3) — a result that a source-audit oracle cannot overturn (G4).
- SRC as a *source-endpoint-control* method: improving the source worst-domain endpoint anti-transferred to
  the target under the tested configuration (G5).

**What is retained.**
- Support-aware extractable leakage `L_Q^ov` as a **measurement object** (estimable-cell restriction,
  grouped cross-fit, held-out source audit).
- The pre-registered K1 (multiplicity-controlled held-out leakage) and K2 (reproducible worst-domain endpoint)
  gates as **decision instruments**.
- **Source-audit oracle replay** as an escape-hatch diagnostic that separates "the control signal is
  uninformative" from "the selector or split was bad."
- **Source→target anti-transfer analysis** as a falsification diagnostic for source-side endpoint objectives.

**What is not claimed.** We do not claim that domain generalization fails, that EEG transfer is impossible,
that support-aware invariance is useless, or that support mismatch is empirically established by BNCI2014-001
alone. We do not claim generality across datasets or backbones, and we do not claim that every DG penalty (or
every well-regularized source-robust objective) must fail. The contribution is the falsification battery and
the two case studies it localizes, not a universal negative.

**Why a measurement-first, falsification-first stance is useful.** Source-side control signals are attractive
because they are computable without target labels; that is exactly why they are easy to trust prematurely. A
battery that audits such signals against held-out target endpoints — before a method is believed — turns a
plausible-but-untransferring penalty into a reported, reproducible failure rather than a published gain. The
same instruments would, in principle, *certify* a genuinely transferring method; demonstrating that
(discriminative validity) is the natural next step.

## Limitations

We foreground the limitations that an adversarial review of our own claims surfaced; each is scoped so the
main text does not lean on it.

1. **One dataset family, one backbone.** All results are BNCI2014-001 LOSO with ShallowConvNet. We do not
   claim the measurements or verdicts are dataset- or backbone-invariant.
2. **Support-mismatch existence is not quantified here.** BNCI2014-001 is balanced 4-class motor imagery; we do
   not report a count of estimable vs. unsupported cells, so we do not claim that a support-mismatch regime is
   empirically exercised on this data. The support-aware construction is motivation and measurement apparatus,
   not a demonstrated property of this dataset.
3. **No naive-vs-support-aware contrast.** We do not present a naive (ungrouped / support-agnostic) diagnostic
   giving a spurious answer that the support-aware `L_Q^ov` corrects; "ill-posed under mismatch" is a premise.
4. **Probe-relativity.** `L_Q^ov` is defined relative to a fixed probe-capacity family and reference prior;
   we do not vary them, so the measurement's sensitivity to that choice is unquantified.
5. **The oracle is a source-audit oracle.** G4 rules out rescue *from held-out source signal*; it is not a
   target oracle, so it does not establish that no rescuing checkpoint exists in an absolute sense.
6. **SRC anti-transfer is single-seed and un-swept.** The anti-transfer result is seed-0 only, n = 6, with no
   confidence intervals, and only the smoothing temperature was varied. The source-guard NLL collapse to
   ≈ 0.09 indicates guard memorization, so anti-transfer under a *well-regularized* source objective is
   untested — the most important open question for this result.
7. **No positive control / discriminative validity.** The battery has only ever returned `falsified`; we have
   not run an ERM-beating method through it, so its ability to certify success (not merely flag failure) is
   unshown.
8. **Minimum-seed / paused protocol.** K1/K2 use seeds [0,1,2] (a minimum-seed configuration, not a full
   5-seed manifest), and BNCI2014-004 / additional seeds are deliberately not run under a pre-registered pause.

The reviewer-objection matrix (supplementary) lists these together with the specific committed numbers that
do and do not answer each objection.

## Figures and tables

See `C16_FIGURE_TABLE_PLAN.md` — 5 figures and 6 tables, *specified not generated*; every quantity is committed to branch `oaci` and regenerable via `oaci/confirmatory/c15_spine.py` and `oaci/falsification/report.py`.
