# C85TR1 Overall Report

## Disposition

```text
C85T_EXECUTION_GUARD_RNG_REPLICATE_PERSISTENCE_AND_PROOF_REVIEW_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION
```

C85TR1 replaces the non-operative C85TL execution lock with one prospective
C85T V2 path that is fail-closed on authorization reuse, RNG-byte drift,
aggregate-only persistence, static execution tokens, same-process proof
approval, and incomplete lifecycle evidence.

C85TR1 itself performs no registered scientific execution. The repository
contains no V2 authorization record, authorization-consumption receipt,
registered S0-S10 result, canonical proof candidate, independent proof verdict,
or theorem-status transition.

## Authorization Disposition

The implementation request ended with `授权 C85T`, but that statement was
received before the C85TR1 protocol, implementation, V2 lock SHA, V2 lock
commit, authorization ID, and exact output root existed. It cannot be bound to
the unique V2 object required by the repair protocol.

Therefore:

```text
pre-lock statement recorded as V2 authorization:
  false

V2 authorization record created:
  false

authorization consumed:
  false

statement reusable after lock:
  false
```

A new post-readiness direct statement is required.

## Authoritative Identities

| Object | Commit / SHA-256 |
|---|---|
| accepted C85TL starting HEAD | `2bebc86f9b42c29f4982b27cc619250948e382b4` |
| C85TR1 protocol commit | `46442b281d61d00a575fae17685648b749659263` |
| C85TR1 protocol SHA-256 | `9c0a7084a7ddd83ef96b8d7f95faf89138829729c0acc5c3d6baeb0ef87ab13d` |
| C85TR1 implementation commit | `f17e25d0d8dc117f7973f90743e07139eeb0c1e1` |
| C85T V2 lock commit | `920c5540a6ae157b77f2acb36f227bfdc172110b` |
| C85T V2 lock SHA-256 | `0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719` |
| V2 runtime registry SHA-256 | `7a52c6c82b81e65f1fb8872705e239b51925d89c955c2145fc18f1fea244919d` |
| historical lock SHA-256 | `4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991` |

Chronology is protocol, then implementation, then lock. All three commits were
pushed before final reporting. The lock binds implementation commit
`f17e25d0...` and 133 repository objects. A post-lock audit found no retained
byte drift.

## Foundation Replay

```text
C85P protocol:
  af4c2cb35a6b6555d6c9ded3105eb7ad4f061ba237d3e8cc3ed6f5a18aede006

C85R repair protocol:
  e37bb444fdd174ba4ca1f95e91d9193378f11dd0ef2aeac3e03cbf6249a34b68

C85R V2 generator:
  e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a

C85TL operationalization:
  6543d6ebbfccb8158f8f48a4fe6409c6243a708bbb0358d350932dd249e6b7c2
```

C85TR1 changes none of the registered state laws, utilities, policies,
scenario modes, sample sizes, estimands, theorem statements, or scientific
boundaries.

## Historical Lock Preservation

The historical C85TL lock remains at its original path and exact SHA. Its
supersession ledger records:

```text
authorization record:
  absent

authorization consumption:
  0

registered execution:
  0

status:
  SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION
```

It is retained as evidence of the seven pre-authorization blockers. It is not
used as an operative fallback.

## Repair 1: S9 RNG Byte Identity

The operative S9 generator now follows the existing protocol exactly:

```text
dtype requested from Generator.integers:
  numpy.int64

draw order:
  51 L then 46 H

mapping:
  0 -> -1
  1 -> +1

canonical byte dtype:
  <i8
```

The old uint8 path remains shadow-only so historical bytes can be reproduced.
Readiness tests establish that uint8 and int64 streams differ in both values or
byte layout; V2 never silently treats them as equivalent.

The frozen S9 digest registry in a future result must contain 4,096 rows and
the L, H, and combined SHA-256 for each mapped int64 replicate stream.

## Repair 2: Probability Intervals

V2 reports two intervals for probability estimands:

```text
raw:
  p_hat +/- 1.96 * MC_SE

reported:
  clip(raw, 0, 1)

flag:
  interval_clipped
```

The point estimate is not modified. Mean regret uses an unbounded raw interval
and is never clipped. Both boundary and interior cases are covered by shadow
tests.

## Repair 3: Replicate Persistence

The future V2 result must persist deterministic NPZ files:

```text
S6_replicates.npz:
  4,096 rows

S7_replicates.npz:
  4,096 rows

S9_replicates.npz:
  8,192 design rows
```

Required S6/S7 arrays are replicate ID, selected action, top-1,
outside-near-set, and regret. S9 persists replicate ID, selected action,
correct-best, top-2 coverage, regret, and `D_hat` for both passive and Neyman
designs, plus paired endpoint arrays.

The writer uses deterministic uncompressed NPZ members with fixed metadata and
forbids object or nonfinite arrays. It reloads the files and reconstructs every
aggregate before a manifest can publish. Missing, duplicate, reordered, or
wrong-length replicate IDs fail.

Shadow persistence evidence:

```text
S6 schema aggregate replay: exact
S7 schema aggregate replay: exact
S9 schema aggregate replay: exact
registered scenario IDs used: 0
```

## Repair 4: Globally Single-Use Authorization

Future authorization uses schema:

```text
c85t_direct_pi_authorization_record_v2
```

It requires a UUID or 256-bit nonce and binds the direct statement, lock SHA,
lock commit, exact content-addressed output root, and exact external
consumption path. C85E, active acquisition, real data, and manuscript fields
must be false.

The authorization's consumption path depends on the authorization hash. To
avoid an impossible self-referential ordinary file hash, V2 defines a
normalized binding digest by replacing only the path with one fixed schema
marker. The actual path is then derived from that digest. The ordinary file
hash is separately captured in preflight.

Consumption performs one exclusive create:

```text
O_WRONLY | O_CREAT | O_EXCL
```

The receipt binds authorization SHA/ID, lock SHA/commit, exact root, attempt
ID, timestamp, and HEAD. An existing receipt blocks reuse globally, including
an attempted move to another output root. Consumption survives execution
failure.

## Repair 5: Private Runtime Capability

No exported static string is operative. A private frozen capability can be
constructed only with a module-private sentinel and is issued only after the
exclusive receipt succeeds.

The object and process issuance registry bind:

```text
authorization SHA;
lock SHA;
attempt ID;
output root.
```

Registered RNG, exact scenario, Monte Carlo, and proof-candidate paths require
that object. Strings, fabricated objects, `None`, and mismatched attempt/root
bindings fail.

## Repair 6: C85T/C85V Proof Separation

C85T may write seven self-contained proof **candidates** and one candidate
disposition registry. Its internal checks cover schema and internal
consistency only. They are explicitly not an independent review.

Every future C85T theorem row must retain:

```text
historical_status:
  OPEN

formal_status:
  OPEN
```

The old automatic transition path is disabled. Even a fabricated same-process
`PASS` cannot produce a non-OPEN status. C85V is the only future stage allowed
to perform an independent read-only proof audit and status transition, and it
requires separate PM approval.

At C85TR1 completion:

```text
T1: OPEN
T2: OPEN
T3: OPEN
T4: OPEN
T5: OPEN
T6: OPEN
T7: OPEN
```

## Repair 7: Append-Only Lifecycle

The coordinator creates a fresh JSONL lifecycle file before registered work.
It enforces the exact order from preflight through authorization, exact
scenarios, Monte Carlo, proof candidates, manifest, and atomic publication.

Each event has a sequence, timestamp, stage, authorization SHA, lock SHA,
attempt ID, and applicable artifact hash. Every write is append-only and
`fsync`-backed. `FAILED` is terminal and records the last completed stage and
primary exception without replacing it with cleanup noise.

## V2 Runtime Replay

The future coordinator replays before authorization consumption:

```text
lock sidecar;
lock schema and unauthorized readiness status;
C85P/C85R/V2/C85TL/C85TR1 identities;
runtime registry;
133 bound repository objects;
all current SHA-256 and Git blobs;
exact environment files;
branch oaci;
clean worktree;
HEAD == origin/oaci;
lock/implementation ancestry;
fresh authorization;
fresh exact root.
```

The 133 bound objects total 574,330 bytes. The separately bound runtime
registry is 21,775 bytes with SHA
`7a52c6c82b81e65f1fb8872705e239b51925d89c955c2145fc18f1fea244919d`.

## Environment

```text
prefix:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python:
  3.13.7

NumPy runtime:
  2.4.4

NumPy metadata first match:
  2.3.3

bit generator:
  PCG64DXSM

CPU / GPU future envelope:
  1 / 0

RAM / wall / storage envelope:
  8 GiB / 30 minutes / 64 MiB
```

Both observed NumPy metadata trees and eleven imported/source/binary/metadata
files are bound. Environment drift fails before consumption.

## V2 Result Contract

The future result gate is:

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

The manifest requires:

```text
scenario results:                    11
S6/S7 logical replicate rows:     8,192
S9 logical replicate-design rows: 8,192
S9 raw-draw digest rows:           4,096
proof candidates:                      7
formal OPEN statuses:                  7
real-data accesses:                    0
active acquisitions:                   0
```

Publication is one staging-to-final rename after all required paths, counts,
dtypes, hashes, aggregate replays, and protected counters pass. Three shadow
failure points prove that no partial final root is published.

## Entry Point

The single operative future entry point is:

```text
python -m oaci.theory.c85t_execute_v2 run-locked \
  --execution-lock oaci/reports/C85T_EXECUTION_LOCK_V2.json \
  --output-root <EXACT_AUTHORIZED_CONTENT_ADDRESSED_ROOT>
```

The historical entry point cannot perform a proof-status transition and the
historical lock remains non-operative.

## Test Evidence

New C85TR1 tests:

```text
authorization/capability/lifecycle/proof separation: 8
RNG/interval/persistence/atomicity:                  11
lock/chronology/isolation:                            8
total:                                               27
```

Component verification passed all 119 C85-family theory tests. The final
lock-byte replay run passed 27/27.

## Accepted Regression

| Suite | Result | Runtime | stderr |
|---|---|---:|---:|
| focused | 375 passed | 10.44 s | 0 bytes |
| C65 | 986 passed, 1 skipped, 3 deselected | 79.89 s | 0 bytes |
| C23 | 1,397 passed, 1 skipped, 3 deselected | 106.10 s | 0 bytes |
| full | 2,321 passed, 1 skipped, 3 deselected | 315.15 s | 0 bytes |

All accepted stderr files have the empty-file SHA-256. The only skip is the
historical finalized C78F test. Three standing C79P nodes are explicitly
deselected by the committed suite wrapper. `squeue` reported no active
C84/C85/OACI job; `sacct` was not used.

## Red Team

The final red team passed 96/96 checks. It covered chronology, historical
preservation, RNG raw-byte identity, interval semantics, replicate arrays,
authorization replay, capability forgery, proof governance, lifecycle order,
atomicity, byte/blob replay, static imports, regression, scheduler state, and
Git hygiene.

## Protected Boundary

All protected counters remain zero:

```text
registered draws and results;
canonical proof candidates;
independent proof verdicts;
theorem-status transitions;
C85T authorization and consumption;
real project data;
active acquisition;
training, forward, GPU;
C85V and C85E;
new data/model zoo execution;
manuscript work.
```

## Artifact Inventory

```text
repair protocol family:        3 files
new guard/coordinator modules: 2
updated implementation files: 6
readiness/lock builder:         1
contract/runtime tables:       16
new tests:                      3 files / 27 nodes
regression wrapper:             1
V2 lock family:                 2 files
bound repository objects:       133
authorization records:          0
registered result roots:        0
```

No raw data, labels, candidate arrays, model state, checkpoint, optimizer,
cache, or file larger than 50 MiB was added to Git.

## Future Authorization

After PM accepts this readiness result, the shortest valid statement is:

```text
授权 C85T
```

That new statement must be bound to:

```text
lock commit:
  920c5540a6ae157b77f2acb36f227bfdc172110b

lock SHA-256:
  0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719
```

The pre-lock statement does not carry forward. C85V, C85E, active
acquisition, real project data, new data/model zoos, and manuscript work remain
unauthorized.

## Detailed Evidence

```text
oaci/reports/C85TR1_PROTOCOL_READINESS.md
oaci/reports/C85TR1_FINAL_REPORT_RED_TEAM.md
oaci/reports/C85TR1_REGRESSION_VERIFICATION.md
oaci/reports/C85TR1_OVERALL_REPORT.json
oaci/reports/c85tr1_tables/
```
