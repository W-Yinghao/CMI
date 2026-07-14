# C84R3 Protocol Timing Audit

C84R3 is an additive engineering repair designed after C84C job `895366` failed.
The failed run accessed Lee2019_MI EEG and source labels but consumed no target labels,
computed no target scientific metric, and completed no candidate instrumentation unit.

The repair changes only the absolute tolerance for replaying a 1,040-term float32
`z @ W.T + b` operation from `1e-6` to `1e-5`. Softmax, repeat-logit and repeat-z
checks remain at `1e-6`. The model, training plan, candidate IDs, subject partitions,
views, channels, sampling, scientific registry and downstream thresholds are unchanged.

Required chronology:

```text
failed attempt 895366
  < C84R3 repair protocol commit
  < implementation and synthetic calibration
  < canary V4 / field V4 protocol commit
  < C84C execution lock V3 commit
  < fresh direct PI authorization
  < any replacement C84C execution
```

The prior authorization and V2 lock are non-operative for replacement execution. The
failed external root and every partial artifact remain preserved.
