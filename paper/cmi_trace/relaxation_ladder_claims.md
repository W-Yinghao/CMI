# CMI-Trace Relaxation Ladder — claim boundary (do NOT rewrite the main manuscript yet)

This study is **exploratory** (the broad "when does erasure help" question was motivated AFTER the strict P0/P1
results were known). It does not overturn or weaken the confirmed P0/P1 result; it maps the regimes around it.

## Confirmed P0/P1 result (unchanged)
1. All tested domain-invariance objectives reduce measured encoder-CMI.
2. Lower encoder-CMI does not imply lower exact original-head reliance.
3. Target effects modest on BNCI2014_001, null/negative on BNCI2015_001.

## Deterministic verdicts (real; paired subject/fold-cluster 95% CI, seeds together, n_boot 10000)
| dataset | backbone | training | verdict |
|---------|----------|----------|---------|
| BNCI2014_001 | DGCNN graph_z | ERM | INCONCLUSIVE |
| BNCI2014_001 | DGCNN graph_z | encoder-CMI | GENERIC_DIMENSIONALITY_EFFECT |
| BNCI2015_001 | DGCNN graph_z | ERM | NO_POSITIVE_REGIME |
| BNCI2015_001 | DGCNN graph_z | encoder-CMI | INCONCLUSIVE |
| **BNCI2014_001** | **frozen EEGNet** | **ERM** | **TRANSDUCTIVE_POSITIVE** |
| BNCI2015_001 | frozen EEGNet | ERM | **NO_POSITIVE_REGIME** (does NOT replicate) |

### CRITICAL SCOPE — the transductive benefit is SINGLE-DATASET and does NOT replicate
The L2 transductive benefit appears ONLY on **BNCI2014_001 (4-class)** frozen EEGNet. On **BNCI2015_001
(2-class)** frozen EEGNet it does NOT replicate — erasure is strongly HARMFUL at every level (L1 LEACE
−0.063, L2 −0.072, L3 −0.160; LW-LEACE WORSE than same-rank random, specific gains −0.06 to −0.12). A
plausible contributor: LW-LEACE removes the full subject span (rank = n_subjects−1 = 11) out of only 16
EEGNet dims, leaving 5 dims — on a 2-class task this over-removes and destroys the task signal, whereas on
BNCI2014 (8 subjects → rank 8 of 16, 4-class) the removal is less aggressive AND the task direction is more
shared (consistency 0.722). So the beneficial regime is NARROW: frozen encoder + transductive + a
task/dimensionality structure where removing the subject span does not also erase the task. **This is a
fragile, single-dataset positive, NOT a general property of frozen features.**

### HEADLINE — frozen EEGNet (BNCI2014 only) reconciles the FMScope result with our strict result
On the **frozen EEGNet** representation of **BNCI2014_001** (the FMScope-style regime), LW-LEACE subject-axis
erasure:
- **L1 strict source-only DG**: Δ −0.010 [−0.019,−0.001], no better than same-rank random (specific gain
  −0.000 [−0.009,+0.009], beats_random=**False**) → erasure does NOT help under strict source-only DG.
- **L2 target-X-unlabeled (transductive)**: Δ **+0.019 [+0.005,+0.035]** (helps), same-rank random **−0.015**
  (hurts), whitening-only ≈0 → specific gain **+0.034 [+0.020,+0.048]**, beats_random=**True**. A real,
  subject-SPECIFIC, transductive benefit (target Y never used; only the unseen subject's unlabeled geometry).
- **L3 oracle** (cohort-conditioned, subject-grouped CV): LEACE hurts the grouped-CV readout but still beats
  random (specific gain +0.016) — a within-cohort diagnostic, not source→unseen-subject transfer.

**Reconciliation (appropriately bounded)**: FMScope's positive and our strict-DG null can BOTH be correct.
A transductive, subject-specific erasure benefit is POSSIBLE (BNCI2014 frozen EEGNet, L2) — needing (a) a
frozen non-graph encoder, (b) the unseen subject's UNLABELED geometry, and (c) a task/dimensionality
structure that survives removing the subject span. But it is FRAGILE: it does NOT replicate on BNCI2015
(same encoder, 2-class), NEVER holds under strict source-only DG (L1) on any representation, and is ABSENT on
the task-trained DGCNN graph representation. So subject-axis erasure is not a reliable DG tool; where it
helps, it is a narrow transductive effect tied to representation + task structure.

The DGCNN family (below) shows NO beneficial regime; the frozen-EEGNet family shows a TRANSDUCTIVE-only,
subject-specific benefit. This does not weaken the confirmed P0/P1 strict result (L1 is null everywhere).

Evidence: `results/cmi_trace_relaxation_ladder/{protocol_ladder_summary,paired_deltas,gate_decisions,verdict}`.
Statistical unit = held-out subject/outer fold; paired subject/fold-cluster 95% bootstrap; seeds travel
together; n_boot 10000. LW-LEACE = LW-whitened LEACE removing the full centered subject span (rank k−1).

## Manuscript-safe wording (by verdict)
- **BNCI2015 ERM (NO_POSITIVE_REGIME)** + all-strata `beats_random=False`:
  > On the task-trained DGCNN graph representation, subject-axis erasure did not become beneficial under any
  > relaxation examined — a freshly trained readout (L1), access to the unseen subject's unlabeled geometry
  > (L2), or a cohort-conditioned oracle (L3). Informed (LEACE) deletion never beat a same-rank random
  > deletion, so no subject-specific benefit was observed on this representation.
- **BNCI2014 encoder-CMI (GENERIC_DIMENSIONALITY_EFFECT)**:
  > Where a small L2 improvement appeared, LEACE and same-rank random removal improved the readout similarly
  > (specific gain CI includes 0); the apparent effect is explained by generic dimensionality reduction /
  > covariance conditioning rather than subject-specific erasure.
- **Oracle (L3)**: even the cohort-conditioned oracle did not create a source-to-unseen-subject benefit on
  this representation; cohort-conditioned removability is an upper-bound representation diagnostic, not DG.
- **Gates (H5)**: no source-only gate policy beat its identity fallback; the gate is a guarded harm-reducer
  (it correctly refuses erasure where it would hurt), not a positive method.

## Manuscript-safe wording — L2-only positive, SINGLE-DATASET (the frozen-EEGNet finding)
> On one dataset (BNCI2014_001, 4-class), access to the new subject's unlabeled geometry made subject-axis
> erasure beneficial on a frozen encoder representation (transductive L2: LW-LEACE +0.019 target bAcc,
> beating same-rank random by +0.034; subject-specific, not dimensionality/conditioning). This did NOT
> replicate on a second dataset (BNCI2015_001, 2-class, same encoder), where erasure was harmful at every
> level, and it never held under strict source-only DG or on the task-trained graph representation. We
> therefore report it as a fragile, regime- and task-structure-dependent effect, not a reliable method.

## STILL PENDING (honest)
- **TSMNet** frozen-feature ladder: dumps env-blocked (spdnets absent in `icml`; torchaudio ABI break in
  `eeg2025`). EEGNet covers the primary frozen-encoder regime; TSMNet would be a second frozen backbone.
- **Rank sensitivity of LW-LEACE on low-dim / many-subject cells**: the BNCI2015 EEGNet harm is partly a
  rank artifact (removing 11 of 16 dims). A truncated-rank LEACE sweep would separate the identity effect
  from over-removal; the config declares `lw_leace_truncated` as an optional eraser for this.

## FORBIDDEN wording (always)
- "CMI erasure solves EEG DG" / "TOS improves domain generalization".
- "subject erasure generally improves EEG transfer" / "subject erasure generally fails" (utility is
  representation-, protocol-, and task-structure-dependent; the DGCNN negative is one representation, and the
  frozen-feature regime is still pending).
- "oracle-global erasure is source-only DG".
- "fresh-head improvement proves lower original-head reliance".

## Mechanism (why L2 helps on EEGNet but no regime helps on DGCNN)
The Stage-4 source-only diagnostics explain the regime difference:
- **frozen EEGNet**: cross-subject task-direction consistency **0.722** (a strongly SHARED task direction) —
  precisely FMScope's stated condition for erasure benefit. task–subject subspace overlap 0.086 (low, so
  removing the subject axis does not destroy the task).
- **DGCNN graph_z**: task-direction consistency **0.275** (much less shared), overlap 0.047.
A shared task direction (high on EEGNet, low on DGCNN) is what lets transductive subject-axis removal help:
when subjects agree on the task axis, removing the subject-identity axis (using the target's unlabeled
geometry) denoises the readout; when they don't (DGCNN), it does not. This ties the beneficial regime to a
measurable, source-only representation property rather than to erasure per se.
