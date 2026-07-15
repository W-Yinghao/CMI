# C84F Failed Attempt 896185

Authorized C84F job `896185` preserved and froze the complete 1,944-unit model
field, then failed in the target-unlabeled trial-registry collector before any
target artifact or final field manifest was written.

## Failure

The exact exception was:

```text
TypeError: '<' not supported between instances of 'dict' and 'dict'
```

It arose at `c84f_target_instrumentation.py:245`, where the collector attempted
to order raw-file dictionaries with `sorted(dict(row) for row in
view.raw_files)`. The defect is an output-order/schema implementation error. It
does not depend on an EEG value, label, selector score, or scientific outcome.

## Preserved State

- Model units: `1,944 / 1,944`.
- Training phases: `72 / 72`.
- Reused units: `486 / 486`.
- New units: `1,458 / 1,458`.
- Model-field manifest SHA-256:
  `d8931b81a3d68f4b1e098ac6e3ede3cd44cdb6c70cdef9f18a76e0a8c62ecdb2`.
- Target EEG arrays loaded: `118 / 118`.
- Target trial-registry rows frozen: `0`.
- Target artifacts and context slices frozen: `0 / 1,944` and `0 / 76,464`.

Target-y access, construction/evaluation labels, oracle access, selector scores,
scientific statistics, and target scientific metrics were all zero. The failure
therefore caused no outcome contamination.

## Retry Disposition

The authorization was consumed and is not reusable. The failed root, logs,
attempt ledger, partial manifest, complete model-field manifest, and target raw
input manifest remain immutable. No retraining or automatic retry is permitted.

Recovery requires an additive target-stage protocol, an insertion-order-
independent canonical sort, a target-only implementation with no training
callable, a new execution lock, and fresh direct PI authorization.

Gate:
`C84F_TARGET_REGISTRY_CANONICAL_ORDER_REPAIR_REQUIRED_NO_OUTCOME_CONTAMINATION`.

