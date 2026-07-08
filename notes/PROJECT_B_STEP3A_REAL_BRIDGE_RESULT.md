# Project B Step-3A Real-EEG Bridge Result

Dataset **BNCI2014_004**, X shape [2860, 3, 385], classes ['left_hand', 'right_hand'], subjects [1, 2, 3, 4], targets [1, 2], eval_unit subject, status **ok**. Target labels used only **post-hoc**; this is a bridge smoke, **not a full benchmark**.

| target | mode | strict | raw dTTA | coverage | action | ACAR-harm | avoided_harm |
|---|---|---|---|---|---|---|---|
| 1 | in_source_subject_q95 | 0.601 | -0.101 | 1.00 | identity:1 | unavailable | 0.101 |
| 1 | nested_source_subject_excess_q95 | 0.601 | -0.101 | 1.00 | identity:1 | unavailable | 0.101 |
| 2 | in_source_subject_q95 | 0.563 | -0.063 | 1.00 | identity:1 | unavailable | 0.063 |
| 2 | nested_source_subject_excess_q95 | 0.563 | -0.063 | 1.00 | identity:1 | unavailable | 0.063 |

On both held-out targets raw offline TTA was harmful; the router blocked OFFLINE_TTA (ACAR-harm degenerate/unavailable) and accepted support-valid IDENTITY, avoiding the harm. Nested source-subject excess was 0, so nested == baseline on this smoke.
