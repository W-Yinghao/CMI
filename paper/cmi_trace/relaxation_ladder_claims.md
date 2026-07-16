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

### HEADLINE — frozen EEGNet reconciles the FMScope result with our strict result (representation- AND regime-dependent)
On the **frozen EEGNet** representation (the FMScope-style regime), LW-LEACE subject-axis erasure:
- **L1 strict source-only DG**: Δ −0.010 [−0.019,−0.001], no better than same-rank random (specific gain
  −0.000 [−0.009,+0.009], beats_random=**False**) → erasure does NOT help under strict source-only DG.
- **L2 target-X-unlabeled (transductive)**: Δ **+0.019 [+0.005,+0.035]** (helps), same-rank random **−0.015**
  (hurts), whitening-only ≈0 → specific gain **+0.034 [+0.020,+0.048]**, beats_random=**True**. A real,
  subject-SPECIFIC, transductive benefit (target Y never used; only the unseen subject's unlabeled geometry).
- **L3 oracle** (cohort-conditioned, subject-grouped CV): LEACE hurts the grouped-CV readout but still beats
  random (specific gain +0.016) — a within-cohort diagnostic, not source→unseen-subject transfer.

**Reconciliation**: FMScope's positive and our strict-DG null can BOTH be correct. Subject-axis erasure
becomes beneficial only when (a) the representation is a frozen non-graph encoder (EEGNet) AND (b) the eraser
sees the new subject's UNLABELED geometry (transductive L2). It does NOT hold under strict source-only DG on
any representation examined, and it is NOT present on the task-trained DGCNN graph representation at all.

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

## Manuscript-safe wording — L2-only positive (the frozen-EEGNet finding)
> Access to the new subject's unlabeled geometry can make subject-axis erasure beneficial on a frozen
> encoder representation (transductive L2: LW-LEACE +0.019 target bAcc, beating same-rank random by +0.034),
> but the effect does not hold under strict source-only DG (L1 null), and it is absent on the task-trained
> graph representation. The benefit is subject-SPECIFIC (LW-LEACE beats matched-rank random and whitening-only
> controls), not generic dimensionality reduction or conditioning.

## STILL PENDING (honest)
- **TSMNet** frozen-feature ladder: dumps env-blocked (spdnets absent in `icml`; torchaudio ABI break in
  `eeg2025`). EEGNet covers the primary frozen-encoder regime; TSMNet would be a second frozen backbone.
- **BNCI2015_001 × EEGNet** ladder: EEGNet dumps regenerated for BNCI2014_001 only so far; BNCI2015_001
  EEGNet dumps would test whether the transductive benefit replicates on the 2-class dataset.

## FORBIDDEN wording (always)
- "CMI erasure solves EEG DG" / "TOS improves domain generalization".
- "subject erasure generally improves EEG transfer" / "subject erasure generally fails" (utility is
  representation-, protocol-, and task-structure-dependent; the DGCNN negative is one representation, and the
  frozen-feature regime is still pending).
- "oracle-global erasure is source-only DG".
- "fresh-head improvement proves lower original-head reliance".
