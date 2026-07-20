# C86R2 Protocol Timing Audit

## Entry State

```text
entry HEAD == origin/oaci:
  1412067610f3133a5307239307c1f569c337548a

C86R final gate:
  C86_UNTOUCHED_COHORT_AGE_ACCESS_OR_INTERFACE_ELIGIBILITY_RECONCILIATION_REQUIRED

C86R2 protocol SHA-256:
  2e88e2fef7500b12ca8b3c5b19e6aab06df5a7f388781855b73793a1fe75df92
```

## Prospective Boundary

The participant-level adult rule, acceptable evidence hierarchy,
deterministic-subset rule, frozen external catalog universe, all-passing cohort
inclusion rule, common-field requirement, and all final gates were fixed before
any additional participant-level public metadata was opened or any catalog
universe was expanded.

At protocol freeze:

```text
new EEG downloaded/opened before C86R2:       0
new target labels opened:                     0
candidate outputs opened:                     0
active acquisition executed:                  0
new candidate training/forward:               0
registered C86 synthetic results:             0
GPU:                                          0
performance outcomes used:                    false
repair outcome-informed:                      false
```

The known C86R aggregate disposition was not used to select subjects or add a
catalog. The search universe consists of the complete installed MOABB imagery
catalog and the complete first post-protocol EEGDash/NEMAR public catalog
snapshot. No additional catalog may be added inside C86R2.

## Status

```text
PROTOCOL_COMMITTED_BEFORE_ADDITIONAL_PUBLIC_METADATA_INSPECTION
C86LP_C86L_C86D_C86C_F_C86H_NOT_AUTHORIZED
```
