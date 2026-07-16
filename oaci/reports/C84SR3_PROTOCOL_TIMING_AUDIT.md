# C84SR3 Protocol Timing Audit

## Sequence

```text
V4 authorization record commit:
  d699919f8d3b96a2f2e42e94a09df076a9af52d4

V4 job:
  898192

V4 authorization consumed:
  true

V4 Stage-B failure:
  2026-07-16T10:44:20Z

V4 blocker evidence commit:
  e5bddf7535ee898b1e1323293b7f084d450edbd3

C84SR3 protocol authored:
  2026-07-16T10:49:41Z
```

The protocol therefore precedes every C84SR3 implementation change.

## Information Available at Repair Design

The repair used only:

```text
the Stage-B engineering traceback;
construction-view row counts by dataset, target and canonical class;
historical Q0 budget semantics;
the V4 attempt and authorization ledgers.
```

It did not access or compute:

```text
evaluation labels;
candidate evaluation utilities;
selector score vectors;
selected candidates;
regret or top-k endpoints;
Q1/Q2;
label-frontier decisions;
LOTO;
scientific taxonomy.
```

## Availability Decision

Q0 budgets are labels per class. Every Lee2019_MI target has exactly 25
construction labels per class. Consequently, Lee `B=32` is physically
unavailable under the registered without-replacement policy.

The additive repair does not replace, rescale, or impute this budget. It records
Lee `B=32` as input-unavailable, retains Lee `B=16`, retains Cho `B=16/32`, and
keeps the common primary grid `[1,2,4,8,FULL]` unchanged for all datasets.

## Authorization Boundary

Job `898192` consumed the V4 authorization. That authorization cannot migrate.
C84SR3 performs synthetic readiness work only and creates a replacement V5
lock. Any real continuation requires a new direct PI statement bound to V5.
