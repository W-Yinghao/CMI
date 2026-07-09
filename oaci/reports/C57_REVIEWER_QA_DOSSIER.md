# C57 - Reviewer Q&A Dossier

## RQ01
Is this just a negative result?

No. The contribution is a measurement/control separation, rank-gauge mechanism, and information-boundary closure.
Evidence: C31-C56
Boundary: Do not frame as universal DG failure.

## RQ02
Are good checkpoints absent?

No. C31 reports joint-good rate 0.424 and C31/C32 show good candidates are common enough; localization fails.
Evidence: K_C31_joint_good_rate
Boundary: Do not imply endpoint scarcity.

## RQ03
Did target labels leak into selection?

No selected artifact is emitted. Target labels appear only in diagnostic ceilings and endpoint-oracle audits.
Evidence: C48-C56
Boundary: Do not describe diagnostics as deployment.

## RQ04
Why is local Bayes ceiling not an action rule?

It is an upper envelope over label-scored neighborhoods, not a source-measurable rule at selection time.
Evidence: C48-C51
Boundary: Ceiling is diagnostic-only.

## RQ05
Why does conditioning not rescue actionability?

C50/C51 show trajectory fragmentation and existing-score underuse despite broad diagnostic coverage.
Evidence: K_C50_trajectory_fail_fraction;K_C50_max_mean_underuse_gap
Boundary: Do not call grouped diagnostics source-only.

## RQ06
Why does C55 not show a transferable endpoint-template method?

Template-only best is 0.704 and does not beat max null p95 0.771; 0.944 requires held-out endpoint scalar.
Evidence: K_C55_template_only_best;K_C55_max_null_p95;K_C55_endpoint_scalar_transfer
Boundary: Preserve the C55 null clarification.

## RQ07
What exactly is unavailable at selection time?

Target-label-derived diagnostics and same-label endpoint scalar/margin are unavailable under original source-only DG.
Evidence: C52-C55
Boundary: Do not call endpoint scalar an available method.

## RQ08
Is split-label or few-label calibration ruled out?

No. It remains future work because the split-label cache is unavailable in current artifacts.
Evidence: K_C53_split_label_budget_available
Boundary: Do not claim few-label sufficiency.

## RQ09
How does this relate to IRM, DomainBed, invariant DA lower bounds, and the broader literature?

Those works frame invariance/model-selection/lower-bound caution; C57 uses them for claim discipline only.
Evidence: C57 literature
Boundary: No universal lower-bound claim.

## RQ10
What is the contribution if no new method is proposed?

A falsification framework, rank-gauge diagnosis, selector/localization audit, and availability-separated endpoint boundary.
Evidence: CL01-CL16
Boundary: No SOTA or method claim.

## RQ11
What is EEG-specific here?

The observed cross-subject EEG candidate universe shows source-visible rank plus target-specific gauge/offset and endpoint-label availability gaps.
Evidence: C31-C56
Boundary: Do not claim EEG transfer impossible.

## RQ12
What should happen after C57?

If C57-A passes, move to M1 manuscript drafting; only repair named inconsistencies if found.
Evidence: C57 decision
Boundary: Do not open another exploratory C-number without a named gap.
