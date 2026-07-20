# C40 Red-Team Verification

Scope: C40 leakage point drift forensics artifacts and reports.

## Verdict

- Red-team result: **pass with trace-boundary enforcement**.
- C40 does not reopen C39 atom claims.
- Active cases are `D2 + D4 + D5 + D6 + D7`; `D1` and `D8` remain inactive.

## Checks

- Frozen identity gate is unchanged: `1e-9` remains binding for elevated atom claims.
- Selection identity is still failed: `48 / 76` at `1e-9`; max drift is `2.1521578246364026e-4`.
- All selection candidates pass only at the loose diagnostic `1e-3` rung; this does not unblock atom claims.
- Stagewise localization found no committed-artifact support/fold/population/cell-mass mismatch; first observed divergence is persisted aggregate point identity.
- Atom additive identity remains exact for recomputed points, but that is not sufficient to explain persisted C37 aggregate leakage.
- Diagnostic atom patterns are stable under observed drift, but remain blocked.
- Exact recovery requires future per-fold/per-cell leakage trace fields; current artifacts are insufficient.
- All tests were run through Slurm on `cpu-high`, not as heavy login-node work.

## Validation

- Slurm C40 single test job `890047`: `8 passed`.
- Slurm C23-C40 regression job `890048`: `165 passed`.

## Blocked Claims

- No atom mechanism claim is elevated.
- No selector repair, deployment, rescue, source-only detector, or target-unlabeled DG success is claimed.
- No tolerance-ladder result is treated as a replacement for exact persisted identity.
