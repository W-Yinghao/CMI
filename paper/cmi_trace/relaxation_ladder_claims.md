# CMI-Trace Relaxation Ladder — claim boundary (do NOT rewrite the main manuscript yet)

This study is **exploratory** (the broad "when does erasure help" question was motivated AFTER the strict P0/P1
results were known). It does not overturn or weaken the confirmed P0/P1 result; it maps the regimes around it.

## Confirmed P0/P1 result (unchanged)
1. All tested domain-invariance objectives reduce measured encoder-CMI.
2. Lower encoder-CMI does not imply lower exact original-head reliance.
3. Target effects modest on BNCI2014_001, null/negative on BNCI2015_001.

## DGCNN graph_z feature family — deterministic verdicts (real, both datasets complete)
| dataset | training | verdict |
|---------|----------|---------|
| BNCI2014_001 | ERM | INCONCLUSIVE |
| BNCI2014_001 | encoder-CMI | GENERIC_DIMENSIONALITY_EFFECT |
| BNCI2015_001 | ERM | NO_POSITIVE_REGIME |
| BNCI2015_001 | encoder-CMI | INCONCLUSIVE |

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

## STILL PENDING (honest)
- **TOS EEGNet frozen-feature ladder** (the FMScope-style regime): EEGNet dumps regenerating in env `icml`
  (jobs 897813-815). The EEGNet ladder result is the direct test of the FMScope-style positive on a frozen
  non-graph representation and is NOT YET IN. TSMNet dumps are env-blocked (spdnets/torchaudio).
- Until the EEGNet ladder is in, the manuscript may NOT generalize the DGCNN negative to "frozen features."

## FORBIDDEN wording (always)
- "CMI erasure solves EEG DG" / "TOS improves domain generalization".
- "subject erasure generally improves EEG transfer" / "subject erasure generally fails" (utility is
  representation-, protocol-, and task-structure-dependent; the DGCNN negative is one representation, and the
  frozen-feature regime is still pending).
- "oracle-global erasure is source-only DG".
- "fresh-head improvement proves lower original-head reliance".
