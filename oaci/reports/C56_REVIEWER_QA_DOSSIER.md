# C56 - Reviewer Q&A Dossier

## RQ01
Is this just a negative result for one dataset?

No: the positive content is an information-boundary map; the scope remains OACI/EEG-DG artifacts.
Evidence: C31,C42-C55
Boundary: Do not generalize to all DG.

## RQ02
Are good checkpoints absent, or merely not localized?

They are present; C31 shows joint-good candidates, while C42-C43 show weak localization.
Evidence: C31,C42,C43
Boundary: Do not claim absence.

## RQ03
Did target labels leak into action construction?

C56 separates source-available rows from diagnostic target-label rows; C53-C55 endpoint closure is marked diagnostic-only.
Evidence: C52-C55
Boundary: Do not present target-label diagnostics as source-available.

## RQ04
Why is local Bayes ceiling not an action rule?

C48-C50 ceilings use conditioned diagnostic neighborhoods and fragment at trajectory level.
Evidence: C48,C49,C50
Boundary: Ceiling is not an action rule.

## RQ05
Why does conditioning not rescue actionability?

C47 improves conditioned source neighborhoods but remains below reliability; C50 shows trajectory fragmentation.
Evidence: C47,C50
Boundary: Conditioning is a separate problem class.

## RQ06
Why do C54/C55 not prove a target-aware action rule?

C54 is same-label endpoint oracle; C55 full closure reads held-out endpoint scalar.
Evidence: C54,C55
Boundary: Endpoint scalar unavailable under original source-only DG.

## RQ07
What exactly is source-visible and what is target-only?

I1 contains source rank/risk/leakage; I6-I7 use target-label-derived endpoint content.
Evidence: C52-C55
Boundary: Do not mix information classes.

## RQ08
Is cross-cell endpoint-template transfer an escape hatch?

It is partial at 0.704 and does not match 0.944 without held-out endpoint scalar.
Evidence: C55
Boundary: C55 closes full-transfer escape hatch.

## RQ09
Is split-label or few-label calibration ruled out?

No. It is not evaluated because split-label cache is unavailable.
Evidence: C53-C55
Boundary: Leave as future work.

## RQ10
Are the nulls and baselines fair?

C55 clarifies that null pass refers to endpoint-scalar transfer, not template-only 0.704.
Evidence: C55,C56
Boundary: Do not overstate template-only null result.

## RQ11
What literature does this connect to?

DomainBed, IRM, invariant representation lower bounds, post-selection inference, and lower-bound frameworks constrain claim language.
Evidence: C56 literature
Boundary: No broad survey or SOTA claim.

## RQ12
What should the next direction be?

If C56-A passes, move to manuscript/theory scaffold rather than new exploratory C-numbers.
Evidence: C56
Boundary: Only targeted repair if a named inconsistency appears.
