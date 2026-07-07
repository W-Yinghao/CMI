# Project B Identifiability Note

## Proposition
Without a representativeness assumption linking source pseudo-target domains to the deployment target
domain, source-only calibration cannot identify either action harm or identity output error under
arbitrary concept shift.

## Construction
Consider two worlds with identical source data distribution, identical unlabeled target feature
distribution, and identical action diagnostics, but a different target label mechanism (concept
relation):

    same source observations
  + same unlabeled target diagnostics
  + different target label mechanism
  => same router decision, different true target error/harm.

Any source-only router sees identical observable information in both worlds and must take the same
action; it cannot simultaneously control harm/error in both. Therefore both `risk_harm(a, target)` and
`risk_error(identity, target)` are non-identifiable from source-only observables under arbitrary
concept shift.

## Consequences
- `OACI_ACAR_HARM_CALIBRATION_DEGENERATE` is a necessary state, not an implementation failure.
- Refusal-first is the rational default when the risk quantity is non-identifiable.
- Support-valid identity does not guarantee concept correctness.
- ACAR-error can only repair error modes that leave an observable, source-representative signature.

## Empirical status (S0 / HF3 probe)
HF3 (source-representative concept, concept_frac=0.50): 12 source-representative catch, 0 observable-diagnostic catch, 4 boundary-confirmed evasion (of 16 concept-degraded HF3 targets; 12 caught total). H_OOD (target-only concept, concept_frac=0.17): corr(pred,true)=-0.22, toy violation=0.29 -> target-only boundary confirmed. Net: source-only ACAR-error is partially estimable where the error mechanism is source-representative/observable, but reproduces the non-identifiability boundary for target-only concept shift; it does NOT solve concept shift.
