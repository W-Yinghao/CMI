# C84SR2 Protocol Timing Audit

## Trigger And Boundary

Authorized C84S V3 job `897843` consumed its fresh authorization and completed
Stage A. Stage B then failed during field-descriptor enumeration, before the
first selector score, because one historical sidecar schema lacked
`level_intervention_id`.

At repair-protocol creation:

```text
construction label access: 1
evaluation label access:   0
selector contexts:         0
scientific result rows:    0
training / forward / GPU:  0 / 0 / 0
same-label oracle:         0
```

The evaluation descriptor remained sealed from Stage B. No target prediction,
selection, regret, Q1/Q2 result, label frontier or scientific taxonomy was
computed or inspected.

## Root Cause Timing

The complete field manifest has an explicit level-intervention identity for all
1,944 units. Exactly 243 historical C84C sidecars, all reused panel-A/seed-5/
level-0 units, predate the level protocol and omit that redundant field. The
remaining 1,701 sidecars include it. This distribution was established from
manifest and sidecar schemas only, without reading candidate arrays or label
table values.

## Prospective Repair

Protocol SHA-256:

```text
6d7853cd60a85c9f3516cb21fda1c75909f0963e96ad2ac0292647bdc93f1aef
```

The protocol precedes all C84SR2 implementation. It permits only a fail-closed
schema bridge from the frozen complete-field descriptor, with exact provenance,
level, panel and seed constraints for the 243 historical omissions. It changes
no candidate, method, score, threshold, label split or inference rule.

The valid Stage-A views are immutable inputs to a future V4 execution. Repair
readiness will not reload target labels. Job `897843`, its consumed
authorization, failed root and ledgers remain preserved and non-reusable.

Future execution requires a new V4 lock, a fresh output root and a fresh direct
PI authorization.
