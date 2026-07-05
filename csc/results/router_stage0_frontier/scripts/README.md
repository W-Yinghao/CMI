# Router Stage R0 exposed frontier (diagnostic-only)

```
Scope: Router Stage R0 | development-only exposed frontier | not confirmatory | not deployable
  not a concept certificate | no method/h0/h1/statistic/feature change | no oracle feature use
  oracle labels used only for diagnostic annotation (NOT in this R0 -- R0 uses no oracle field at all)
```
Rule (MONOTONE): ALLOW if B3 method_confirm==True AND observed_T>=tau, else ABSTAIN. The allow-set is a strict
SUBSET of B3 confirmations -> can only REMOVE confirmations, never ADD (avoids the B4c clean-false-add regression).
Primary deployable score = observed_T. No oracle field, no condition label in the decision. Deterministic grid.
router_stage0.py re-derives the frontier from the committed csc/results/p3_forensics/p3_internal_forensics_merged.jsonl.
