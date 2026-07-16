# CMI-Trace DG-erasure oracle — execution note + GO/NO-GO

The pivot (post-hoc oracle/TTE showed the removable leakage is functionally UNUSED -> can't help DG). New,
INVERTED objective: minimize SOURCE-HELD-OUT risk (DG-relevant), CMI reduction only a CONSTRAINT. Branch
`agent/cmi-trace-dg-oracle` (base = frozen erasure-oracle results). Env `c84c-eeg2025-v3`.

## Synthetic proof (safe != DG), spurious-task DGP [tests 5/5]
Z=[Z_inv stable | Z_spur source-predictive-but-flips-on-target | Z_id pure subject]. TARGET DG oracle deletes
Z_spur -> target +0.296; SOURCE-META oracle (source-LOSO, source-only) recovers it -> true target +0.280
(Result A: source-identifiable); CMI-only selector +0.129 (WORSE, deletes unused Z_id/Z_inv). Minimizing CMI
!= improving DG.

## REAL EEG go/no-go (EEGNet, both datasets, subject/fold-cluster 95% CI)
| dataset | verdict | target UB (achievable) | source-meta (deployable) | matched-rank random | CMI-only |
|---------|---------|------------------------|--------------------------|---------------------|----------|
| BNCI2014_001 | B (target-only) | +0.016 [+0.011,+0.020] | -0.001 [-0.003,+0.002] | -0.005 | -0.010 |
| BNCI2015_001 | borderline A    | +0.054 [+0.027,+0.098] | +0.002 [+0.001,+0.004] | -0.010 | -0.063 |

### VERDICT: predominantly Result B (near-miss A on one dataset). NOT Result C.
- A target-beneficial linear subject-subspace deletion EXISTS on both datasets (target-label upper bound
  +0.016 / +0.054, CIs exclude 0) -> the candidate dictionary is NOT empty.
- BUT the SOURCE-ONLY meta-selector cannot reliably identify it: source-meta -0.001 (BNCI2014, Result B) and
  +0.002 (BNCI2015, statistically >0 & >random but ~4% of the +0.054 achievable -> borderline/near-miss A).
- CMI-only objective HURTS DG on both (-0.010 / -0.063): minimizing CMI deletes the WRONG (unused) directions.
  Re-confirms safe-erasure != DG-erasure on real EEG.

### Decision (per the pre-registered A/B/C matrix)
Result B => a DG-beneficial subset exists but is NOT source-identifiable from the current candidate basis +
source-LOSO meta-objective. Therefore: **do NOT proceed to a deployable source-only differentiable subspace
supermask (P2) on this basis** -- it would not reliably find the ticket. Honest next steps are (a) a STRONGER
candidate basis (label-conditional subject subspace / per-domain task-gradient-disagreement directions) and/or
a stronger source-meta objective, or (b) TARGET-UNLABELED (transductive) adaptation -- NOT a source-only DG
method claim. The safe-erasure oracle + TTE V1 (cleaning) results are unchanged and remain the confirmed line.
