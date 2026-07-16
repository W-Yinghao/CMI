# DG-level measurement→action gap: a cross-subject-relevant subject-subspace deletion EXISTS but is not source-identifiable

*(Draft section for the CMI-Trace manuscript; continues C12/C13. Verdict verified by a 5-skeptic adversarial
workflow, high-confidence DOWNGRADE_TO_HINDSIGHT_ONLY. The one identifiability CI marked `[[AUDIT_*]]` is
filled from `results/cmi_trace_dg_identifiability/source_greedy_audit_verdict.json`.)*

## Motivation — was the objective wrong, not the method?

C12/C13 established that no source-fitted eraser built to reduce measured conditional subject leakage
`I(Z;D|Y)` yields a practically meaningful target-bAcc gain in any valid dataset–backbone cell. A natural
objection: those erasers minimize *leakage*, whereas domain generalization asks to minimize the risk
contributed by the *subject-unstable, task-bearing* part of the representation. The exact-head-null oracle
sharpens this — **on the task-trained DGCNN graph representation (which stores a replayable linear task head),
most measured conditional subject leakage lies in the exact nullspace of that head** (functionally **unused**),
so removing it is provably task-safe yet cannot change the decision. (The DG-identifiability experiments below
use the EEGNet frozen representation, which has no replayable original head; there the head-defined "contested"
subspace uses a fresh source-fitted linear head.) Safe removability and DG relevance are therefore different
objects a priori. This section asks the DG-relevant question directly, with the objective inverted:
**minimize source-held-out risk** (source-LOSO), with subject-CMI demoted from objective to post-hoc
certification constraint.

## Two questions, two oracles — and why the oracle must be mechanism-matched

- **Existence (upper bound).** A *cross-fitted* target oracle selects a deletion subset on a target split
  `T_select` (target labels — non-deployable) and reports its gain on a **disjoint** `T_query`. Cross-fitting
  is essential: selecting and scoring on the same target trials inflates the gain (subset-search optimism).
  The selection must be **mechanism-matched**: a differentiable subspace supermask deletes an *arbitrary
  coordinate set*, so the honest existence test is greedy arbitrary-coordinate selection, not top-k of an
  ordered basis. (Restricting to top-k *prefix* deletion is a strictly weaker oracle and can spuriously
  report ~0 — a pitfall we hit and corrected.)

- **Source identifiability (the deployable question).** A greedy source-only selector over **an
  outer-source-fitted basis with source-LOSO utility selection**: the candidate basis is estimated from all
  outer-source subjects, then directions are added greedily to maximize source-leave-one-subject-out held-out
  bAcc (each source subject's own head-fit excludes it, but its features do enter the fixed basis estimate —
  so this is *not* a fully-nested meta-validation; the information advantage only makes the source selector
  more favorable, so a negative result is conservative). The true outer target is never used in selection.
  The selected deletion is applied to the true target; identifiability requires its target gain to be
  positive, beat matched-rank random, and align (principal angles) with the greedy target ticket. (The
  separate refittable *prefix* rule of the nested-meta selector is the only fully-nested construction here.)

Candidate bases: marginal subject span (`marg`), label-conditional subject offsets (`cond`, aligned to
`I(Z;D|Y)`), decision-rule disagreement (`rule`), task-gradient disagreement (`grad`); each also restricted
to the **contested** subspace (row space of the class-centered source head — directions the head uses; the
`ker(W_c)` complement is the functionally-free leakage the exact-head-null oracle already handles).

## Synthetic anchor — the machinery is honest in both directions

On a spurious-task DGP (`Z=[Z_inv|Z_spur|Z_id]`, `Z_spur` predictive-within-source but sign-unstable across
subjects; full table in `notes/DG_SYNTHETIC_SELECTOR_TABLE.md`): the cross-fit greedy oracle finds the real
ticket (deleting `Z_spur`) beating random; the greedy source-only selector **recovers** it both when the
shortcut is source-visibly *balanced* (Δ_src +0.032 beating random −0.033) and when it is a *strong
majority* shortcut (Δ_src +0.150, subspace alignment 1.0) — the greedy source selector is strictly more
expressive than the nested-prefix rule (which correctly *refuses* the majority shortcut under its no-harm
gate). Because the *stronger* greedy source selector recovers these synthetic tickets, its failure on real
EEG is genuine source-unobservability, not a dead selector. (The proxy-reduction selector instead picks
functionally-unused `Z_id`/`Z_inv` and is worse for DG.)

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
which can, still fails. A **selector that maximizes reduction of a linear within-label subject-decodability
proxy** (the cheap search-time proxy, not the validated posterior-KL ruler) is harmful on both datasets
(−0.010 / −0.063, LCB < 0).

Subject-leakage certification (CORRECTED cross-fit protocol: cond basis fit on the source eraser-fit split
disjoint from the posterior train/eval trials; split-specific greedy ticket applied identically; paired
ΔÎ_specific = mean_i ΔÎ(random_i) − ΔÎ(ticket) at capacities {linear, small, large}, subject-cluster 95% CI;
`cmi_cert2_*.jsonl`): the ticket is **NOT certified** as a subject-leakage-specific deletion on either
dataset — at the primary (high-capacity) critic ΔÎ_specific = −0.007 [−0.035,+0.027] (BNCI2014) and −0.011
[−0.032,+0.011] (BNCI2015), both LCB < 0; the ticket removes **no more validated conditional subject leakage
than a matched-rank random deletion** (ticket +0.040 vs random +0.047; ticket +0.025 vs random +0.036). So
the ticket is definitively **a deletion within a subject-derived basis**, and its within-target DG benefit is
**not attributable to validated subject-leakage removal** — a sharp further instance of *leakage amount ≠ DG
relevance*. (The earlier "1/4 cells certified" pass was an artifact of certifying a re-selected full-target
ticket without the eraser-fit split or a paired control; it is retracted.)

## The claim boundary

> On real EEG a subject-subspace deletion that improves within-target generalization **exists** (cross-fitted,
> above matched-rank random, mechanism-matched greedy selection), yet a source-only selector — including the
> greedy selector that matches the deletion mechanism exactly — does **not** recover it, and **selecting
> deletions by maximal reduction of a linear within-label subject-decodability proxy** is harmful on both
> datasets. The gap is one of **source observability**, established for the tested bases and selectors — not a
> universal impossibility theorem.

The measurement→action chain has four distinct links, and each real-EEG result breaks the *next*:

**leakage amount ≠ safe removability ≠ DG relevance ≠ source identifiability.**

Leakage is measurable (C1–C11); much is safely removable but functionally unused (exact-head-null oracle);
the DG-relevant part is a different, subject-unstable, task-bearing object that provably **exists** (this
section's oracle); and even that part is, on the tested bases and selectors, **not identifiable from source
subjects alone**. This scopes the negative precisely: a source-only differentiable supermask is not warranted
(it would inherit the unidentifiability), and the admissible next step is an explicit change of information
regime — a target-X observability audit (does any *unlabeled-target* statistic predict the ticket?) before
any target-unlabeled adaptation — not a source-only DG method claim.
