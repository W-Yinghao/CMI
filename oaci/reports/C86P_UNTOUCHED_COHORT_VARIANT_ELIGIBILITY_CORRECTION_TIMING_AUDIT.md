# C86P Untouched-Cohort Variant Eligibility Correction Timing Audit

The main C86P protocol remains immutable. During the metadata-only readiness
audit, before any new EEG, label, active-policy, or registered synthetic access,
the installed MOABB summary catalog was compared with loader-source metadata.

The catalog combines the `Yang2025` two-class and three-class variants, while
the canonical default loader is the untouched 51-subject two-class left/right
hand interface. The initial pass therefore omitted a cohort that satisfies the
already locked rule. This additive protocol corrects only variant recognition
and the resulting all-eligible cohort set.

At correction time:

```text
new EEG downloads or opens:        0
new target-label opens:             0
active-acquisition executions:      0
registered C86 synthetic results:   0
C86 scientific results:             0
performance outcomes used:          0
```

Budgets, methods, estimators, endpoints, inference, taxonomy, and claim
boundaries are unchanged. The correction is prospective and metadata-only.
