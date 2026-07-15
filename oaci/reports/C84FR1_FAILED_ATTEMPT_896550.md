# C84FR1 Failed Target-Stage Attempt 896550

Job `896550` consumed the fresh C84FR1 authorization and successfully replayed
the frozen 1,944-unit model field without training. It loaded and validated all
118 target-unlabeled subjects, reproduced the historical raw-input manifest
exactly, and froze a 9,621-row label-free target trial registry.

The job then stopped during complete target instrumentation. The persisted
linear replay error for unit `c84l1_00cb2c89efa87efe281dbb9229c63e53`
(Cho2017, panel B, seed 6, level 1, SRC epoch 149) was
`2.193450927734375e-05`, above the locked `2e-05` gate. The threshold was not
widened and no retry was launched.

At stop, five target artifacts and 268 candidate-context slices had completed;
the failing sixth NPZ exists without a context index. No complete-field manifest
was published. These partial target artifacts are evidence only and are not
automatically reusable. The complete frozen model field remains unchanged.

Protected counters remained zero for model retraining, target-y access, target
label fields, selector scores, scientific metrics, same-label oracle, and C84S.
The authorization is consumed and cannot be reused. Scheduler completion was
tracked with `squeue` plus the application failure ledger; `sacct` was not used.

Gate:

```text
C84F_TARGET_INSTRUMENTATION_NUMERICAL_GATE_REPAIR_REQUIRED
```

Disposition:

```text
FAILED_PRESERVED_NO_LABEL_OR_SCIENTIFIC_OUTCOME_CONTAMINATION
PM_REVIEW_REQUIRED
NO_AUTOMATIC_RETRY
```
