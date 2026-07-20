# C84C Failed Engineering Attempt 895366

Job `895366` consumed the direct C84C authorization and executed the locked V3
engineering canary. It failed closed after the three Lee2019_MI training phases and
before the first complete instrumentation unit.

```text
state / exit:                    FAILED / 1:0
runtime:                         00:34:34
real EEG views materialized:     3
source label arrays read:        2
target-y accesses:               0
target scientific metrics:       0
training phases:                 3 / 3 Lee phases
optimizer states preserved:      81
candidate checkpoints preserved: 1
complete units:                  0
source/target artifacts:         0 / 0
```

The failure was a float32 engineering identity check. CUDA `Linear` logits and a CPU
matrix reconstruction of the same 1,040-term dot product differed by
`2.86102294921875e-06`; softmax, repeat logits and repeat representation errors were
exactly zero. The V3 implementation applied one absolute `1e-6` tolerance to all four
objects, so it rejected the linear reconstruction before writing source or target
instrumentation.

The output root, consumed authorization, attempt ledger, 81 optimizer states, first
checkpoint, stdout and stderr are preserved. The run is not reusable and cannot be
retried under the consumed authorization or historical execution lock. Cho2017 and
PhysionetMI were not entered.
