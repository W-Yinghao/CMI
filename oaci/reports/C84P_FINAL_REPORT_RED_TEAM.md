# C84P Final Report Red Team

## Result

```text
checks:                    65
audit checks passed:       65
audit check failures:       0
open scientific blockers:   1
```

The red team confirms that the implementation correctly stops the proposed
21-channel program: `Lee2019_MI` exposes 20 of the 21 requested channels and
does not contain `FCz`. `Cho2017` and `PhysionetMI` expose all 21. The protocol
forbids interpolation, `Fz` substitution, or silent montage reduction, so the
channel mismatch is a blocking scientific-input condition, not a test failure.

The four protocol hashes replay. C84C, C84F and C84S are explicitly marked
`BLOCKED_NOT_READY_FOR_AUTHORIZATION`; no scope-specific execution lock exists.
The PI's broad authorization intent is recorded, but no future-stage
authorization was consumed because no unique executable lock exists and the
required prior gate did not pass.

All deterministic subject partitions, source/target overlaps, task/event
mappings, half-open 480-sample epoch rule, candidate arithmetic, selector
registry, budget grids, physical views, target-cluster dependence rules,
same-method taxonomy and resource envelopes replay. The S0-S14 synthetic suite
and five auxiliary max-T/Q1/Q2 checks pass without importing a dataset loader.

No EEG array, real label, external dataset payload, model state, candidate unit,
selector result or manuscript experiment was created. No GPU job ran. The same-
label oracle and BNCI2014_004 remain untouched. All four CPU regressions passed
with empty stderr and no C84 path skipped.

The exact checks are in `c84p_tables/final_report_red_team.csv`. The red-team
outcome requires the reconciliation gate; it does not support canary readiness.
