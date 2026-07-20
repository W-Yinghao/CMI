# C79E Repair 002 - Target-4 Instrumentation Worker Binding

## Failure

CPU job `893361` prepared the physically isolated target-4 source and
target-unlabeled input views, then failed before forward execution. Spawned
workers re-imported the historical C78F module and recovered its default
remaining-target registry, which excludes engineering canary target 4.

```text
source input rows loaded:            4,608
target-unlabeled input rows loaded:    576
target labels read:                      0
forward rows:                            0
unit caches created:                     0
model-specific outcomes read:            0
```

## Repair

The locked CLI already exposes `--workers`. The replacement uses `--workers 1`
and `--threads-per-worker 48`, causing `_worker` to run in the authorized parent
process where the seed-4 binding is intact. No file, implementation hash,
scientific registry, model, threshold, null, or lock changes.

This mode is limited to the target-4 engineering canary. Primary Wave A/B
targets are members of the historical remaining-target registry and retain the
locked four-worker mode.
