# DG-level measurement→action gap: a cross-subject-relevant subject-subspace deletion EXISTS but is not source-identifiable

*(Draft section for the CMI-Trace manuscript; continues C12/C13. Verdict verified by a 5-skeptic adversarial
workflow, high-confidence DOWNGRADE_TO_HINDSIGHT_ONLY. The one identifiability CI marked `[[AUDIT_*]]` is
filled from `results/cmi_trace_dg_identifiability/source_greedy_audit_verdict.json`.)*

## Motivation — was the objective wrong, not the method?

C12/C13 established that no source-fitted eraser built to reduce measured conditional subject leakage
`I(Z;D|Y)` yields a practically meaningful target-bAcc gain in any valid dataset–backbone cell. A natural
objection: those erasers minimize *leakage*, whereas domain generalization asks to minimize the risk
contributed by the *subject-unstable, task-bearing* part of the representation. The exact-head-null oracle
sharpens this — on real EEGNet the safely-removable subject leakage sits largely in the task head's
nullspace (functionally **unused**), so removing it is provably task-safe yet cannot change the decision.
Safe removability and DG relevance are therefore different objects a priori. This section asks the
DG-relevant question directly, with the objective inverted: **minimize source-held-out risk** (source-LOSO),
with subject-CMI demoted from objective to post-hoc certification constraint.

## Two questions, two oracles — and why the oracle must be mechanism-matched

- **Existence (upper bound).** A *cross-fitted* target oracle selects a deletion subset on a target split
  `T_select` (target labels — non-deployable) and reports its gain on a **disjoint** `T_query`. Cross-fitting
  is essential: selecting and scoring on the same target trials inflates the gain (subset-search optimism).
  The selection must be **mechanism-matched**: a differentiable subspace supermask deletes an *arbitrary
  coordinate set*, so the honest existence test is greedy arbitrary-coordinate selection, not top-k of an
  ordered basis. (Restricting to top-k *prefix* deletion is a strictly weaker oracle and can spuriously
  report ~0 — a pitfall we hit and corrected.)

- **Source identifiability (the deployable question).** A **greedy source-only** selector that maximizes
  source-LOSO held-out bAcc over arbitrary coordinates — the exact mechanism a supermask implements, driven
  by source data only — is applied to the true target. Identifiability requires its target gain to be
  positive, beat matched-rank random, and align (principal angles) with the greedy target ticket.

Candidate bases: marginal subject span (`marg`), label-conditional subject offsets (`cond`, aligned to
`I(Z;D|Y)`), decision-rule disagreement (`rule`), task-gradient disagreement (`grad`); each also restricted
to the **contested** subspace (row space of the class-centered source head — directions the head uses; the
`ker(W_c)` complement is the functionally-free leakage the exact-head-null oracle already handles).

## Synthetic anchor — the machinery is honest in both directions

On a spurious-task DGP (`Z=[Z_inv|Z_spur|Z_id]`, `Z_spur` predictive-within-source but sign-unstable across
subjects): the cross-fit greedy oracle finds the real ticket (deleting `Z_spur`) beating random; the greedy
source-only selector **recovers** it when the shortcut is source-visibly unstable (balanced sign,
Δ_src +0.032 beating random −0.033; and a strong majority shortcut Δ_src +0.150, subspace alignment 1.0);
and the old CMI-minimizing selector picks functionally-unused `Z_id`/`Z_inv` and is worse for DG. So the
audit *can* say yes; its "no" on real EEG is a finding, not a dead selector.

## Real EEG — the verified verdict: TARGET_HINDSIGHT_ONLY

Existence (greedy cross-fit oracle, cond/full basis, subject/fold-cluster 95% CI):

| dataset | un-cross-fit (optimistic) | **greedy cross-fit oracle** | matched random | prefix oracle (weaker) |
|---|---|---|---|---|
| BNCI2014 | +0.049 [.040,.060] | **+0.021 [+0.012,+0.032]** | −0.015 | +0.005 [−0.000,.012] |
| BNCI2015 | +0.065 [.030,.113] | **+0.045 [+0.010,+0.094]** | −0.007 | +0.001 [−0.003,.006] |

So ~half the apparent gain was subset-search optimism (removed by cross-fitting), but **the other half
survives with LCB > 0 and clearly beats matched-rank random** (negative). A cross-subject-relevant subject-
subspace deletion *does* exist. Two caveats the verification requires: (i) the ticket is **within-target
hindsight** — the oracle selects on `T_select` and scores on the same subject's disjoint `T_query`; it proves
the deletion is real, not that a *source-chosen* deletion transfers to a new subject; (ii) the BNCI2015
signal is **concentrated in 1–2 outlier subjects** (one fold +0.26), whereas BNCI2014 is distributed (8/9
folds positive) and is the robust one.

Identifiability (greedy source-only audit — mechanism-matched, source-LOSO, applied to the true target;
subject/fold-cluster 95% CI): across all four full bases the source-greedy selector does **not** recover the
ticket. On **BNCI2014** every basis has source→target Δ with LCB ≤ 0; the basis with the *largest* oracle
ticket (`cond`, +0.021) is recovered as **−0.0029 [−0.0055,−0.0007]** — source selection is slightly
*anti-aligned* (mean principal-angle cos 0.30). On **BNCI2015** only the marginal basis is statistically
positive (**+0.0024 [+0.0013,+0.0036]**, beating matched-rank random −0.007), but that recovers only ≈7% of
that basis's oracle gain (+0.035), does **not** replicate on BNCI2014, and has low subspace alignment (0.42):
`SOURCE_DETECTABLE_TINY`, not practical (RecoveryRatio ≈ 0.07 ≪ 0.25). The earlier nested selector was
prefix-only and could not even *express* the greedy ticket; the mechanism-matched greedy source selector,
which can, still fails. Directly minimizing conditional subject leakage (CMI-only) is harmful on both datasets
(−0.010 / −0.063, LCB < 0).

## The claim boundary

> On real EEG a subject-subspace deletion that improves within-target generalization **exists** (cross-fitted,
> above matched-rank random, mechanism-matched greedy selection), yet a source-only selector — including the
> greedy selector that matches the deletion mechanism exactly — does **not** recover it, and directly
> minimizing conditional subject leakage is harmful. The gap is one of **source observability**, established
> for the tested bases and selectors — not a universal impossibility theorem.

The measurement→action chain has four distinct links, and each real-EEG result breaks the *next*:

**leakage amount ≠ safe removability ≠ DG relevance ≠ source identifiability.**

Leakage is measurable (C1–C11); much is safely removable but functionally unused (exact-head-null oracle);
the DG-relevant part is a different, subject-unstable, task-bearing object that provably **exists** (this
section's oracle); and even that part is, on the tested bases and selectors, **not identifiable from source
subjects alone**. This scopes the negative precisely: a source-only differentiable supermask is not warranted
(it would inherit the unidentifiability), and the admissible next step is an explicit change of information
regime — a target-X observability audit (does any *unlabeled-target* statistic predict the ticket?) before
any target-unlabeled adaptation — not a source-only DG method claim.
