# Results

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
