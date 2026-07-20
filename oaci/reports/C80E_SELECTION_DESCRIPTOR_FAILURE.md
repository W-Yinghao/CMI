# C80E Selection Descriptor Failure

Authorized job `894641` passed the repaired authorization and all field/view
manifest checks. It completed and froze the construction-only nested-Q0
selection payload, then stopped before opening the evaluation view.

```text
job state:                    FAILED
exit code:                    1:0
runtime:                      00:07:03
selection manifest frozen:   yes
selection payload inspected: no
evaluation-label reads:      0
evaluation outcomes:         0
oracle accesses:             0
target4 primary rows:        0
```

The generic C74 shard verifier assumes every array in an NPZ has the same
first dimension. The registered C80 selection schema intentionally contains
32-cell arrays and one seven-element budget-label array, so the verifier
reported lengths `[7, 32]` and stopped.

This is a descriptor ABI defect, not a scientific result or a reason to alter
the analysis. The frozen selection payload remains content-addressed. Its
values were not opened or inspected during triage.

The additive repair is limited to a selection-specific verifier with exact
registered fields and shapes. It must reuse the existing selection freeze,
must not change the scientific registry, and must receive a new adapter hash
and analysis lock before evaluation resumes.
