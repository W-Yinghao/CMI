# C85TR1 Execution Guard, RNG Persistence, And Proof Review Readiness

## Final Gate

```text
C85T_EXECUTION_GUARD_RNG_REPLICATE_PERSISTENCE_AND_PROOF_REVIEW_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION
```

C85TR1 additively repairs the non-operative historical C85T execution path.
It creates one byte-bound V2 lock, but it does **not** authorize or execute
C85T. No registered S0-S10 stream, canonical proof candidate, theorem-status
transition, real-data path, active acquisition, C85V, C85E, or manuscript path
was opened.

The direct statement appended to the C85TR1 implementation request preceded
the V2 lock and its unique lock SHA/commit/output-root binding. It therefore
cannot satisfy the V2 authorization schema, was not recorded, was not
consumed, and is not reusable. A new direct statement is required after this
readiness gate.

## Authoritative Chronology

```text
accepted C85TL starting HEAD:
  2bebc86f9b42c29f4982b27cc619250948e382b4

C85TR1 protocol-before-implementation commit:
  46442b281d61d00a575fae17685648b749659263

C85TR1 repair protocol SHA-256:
  9c0a7084a7ddd83ef96b8d7f95faf89138829729c0acc5c3d6baeb0ef87ab13d

C85TR1 implementation commit:
  f17e25d0d8dc117f7973f90743e07139eeb0c1e1

C85T V2 execution-lock commit:
  920c5540a6ae157b77f2acb36f227bfdc172110b

C85T V2 execution-lock SHA-256:
  0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719

V2 runtime-bound registry SHA-256:
  7a52c6c82b81e65f1fb8872705e239b51925d89c955c2145fc18f1fea244919d
```

The protocol was committed and pushed before implementation or shadow
calibration. The implementation was committed and pushed before lock
materialization. The lock binds the implementation commit and discovers its
own containing Git commit at future runtime; that future authorization must
bind both the lock SHA-256 and discovered lock commit.

Post-lock byte audit confirmed that all 133 bound objects still match the lock
by path, size, SHA-256, and Git blob. No implementation edit was retained after
lock creation.

## Preserved Historical Objects

The historical C85TL lock remains byte-identical:

```text
path:
  oaci/reports/C85T_EXECUTION_LOCK.json

commit:
  9d414ebb889b2cfc3fefa19fa98d7ea5ca9fd691

SHA-256:
  4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991

authorization:
  absent

registered execution:
  0

operative status:
  SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION
```

The C85P protocol, C85R repair protocol, C85R V2 generator, C85TL
operationalization, historical lock, C85TL reports, and all earlier registries
are preserved. Nothing was rewritten in place.

## Foundation Identities

```text
C85P protocol SHA-256:
  af4c2cb35a6b6555d6c9ded3105eb7ad4f061ba237d3e8cc3ed6f5a18aede006

C85R repair protocol SHA-256:
  e37bb444fdd174ba4ca1f95e91d9193378f11dd0ef2aeac3e03cbf6249a34b68

C85R V2 generator SHA-256:
  e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a

C85TL operationalization SHA-256:
  6543d6ebbfccb8158f8f48a4fe6409c6243a708bbb0358d350932dd249e6b7c2
```

The S0-S10 laws, exact-versus-Monte-Carlo modes, 4,096 replicate count,
estimands, theorem statements, and T1-T7 entering status remain unchanged.

## Blocker Reconciliation

| Historical blocker | V2 repair | Readiness evidence |
|---|---|---|
| S9 protocol `int64`, implementation `uint8` | Operative path requests `numpy.int64`, preserves mapped Rademacher values as little-endian `<i8`, and digests those bytes | AST/source test plus committed shadow byte registry |
| Probability intervals unbounded | Stores raw Wald and clipped reported intervals with an explicit clipping flag | Boundary and interior shadow tests |
| Aggregate-only MC evidence | Persists S6, S7, and S9 deterministic NPZ arrays; reloads before aggregate publication | Three exact aggregate replay fixtures and V2 manifest validation |
| Static exported execution token | Replaced by process-issued private capability requiring a module-private sentinel | Static string and fabricated-object denial tests |
| Authorization replay/movable root | Normalized binding SHA, exact content-addressed root, external `O_CREAT|O_EXCL` receipt | Same-root and changed-root truth-table tests |
| Same-process proof review | C85T freezes candidates only; all formal statuses remain OPEN; C85V is separate | Transition API refuses even fabricated PASS |
| One-shot lifecycle JSON | Canonical append-only JSONL with ordered stage events and terminal failure | Ordered replay and skipped-stage denial tests |

## Exact S9 RNG Bytes

The operative V2 draw path is:

```text
Generator:
  numpy.random.Generator(numpy.random.PCG64DXSM(seed))

seed:
  low64(SHA256(C85_SYNTHETIC_V1|scenario_id|replicate_id))

draw calls:
  rng.integers(0, 2, size=51, dtype=numpy.int64)
  rng.integers(0, 2, size=46, dtype=numpy.int64)

order:
  51 L, then 46 H

mapping:
  0 -> -1
  1 -> +1

canonical persisted digest dtype:
  little-endian int64 (`<i8`)
```

The historical `uint8` helper remains available only to replay the superseded
shadow bytes. It rejects registered S0-S10 identifiers. The operative helper
contains no `uint8` draw request. Shadow tests demonstrate that the historical
and V2 byte streams differ, which prevents accidental equivalence claims.

No registered S9 seed or draw was opened. The committed replay table contains
only `SHADOW_RADEMACHER_A` and `SHADOW_RADEMACHER_B` fixtures.

## Monte Carlo Interval Contract

For top-1 and outside-`A_epsilon` probabilities, V2 stores:

```text
raw_95pct_mc_interval:
  mean +/- 1.96 * MC_SE

reported_95pct_mc_interval:
  clip(raw interval, 0, 1)

interval_clipped:
  true exactly when either endpoint changed
```

The point estimate is not clipped. Mean-regret intervals remain unbounded raw
intervals because regret is not a probability. A shadow case centered at zero
replays `[-0.196,0.196]` raw and `[0,0.196]` reported; an interior case remains
unchanged.

## Exact Replicate Persistence

Future successful C85T V2 output must include:

| Artifact | Logical rows | Required arrays |
|---|---:|---|
| `S6_replicates.npz` | 4,096 | `replicate_id`, `selected_action`, `top1`, `outside_A_epsilon`, `selection_regret` |
| `S7_replicates.npz` | 4,096 | same S6 schema |
| `S9_replicates.npz` | 8,192 design rows | six arrays per design plus paired endpoint arrays |
| `S9_raw_draw_digest_registry.csv` | 4,096 | replicate ID and L/H/combined `<i8` byte digests |

Canonical dtypes are:

```text
S6/S7 replicate_id:          uint16
S6/S7 selected_action:       uint16
S6/S7 indicators:            uint8
S6/S7 regret:                float64

S9 replicate_id:             uint16 per design
S9 selected_action:          uint8 per design
S9 indicators:               uint8 per design
S9 regret and D_hat:         float64 per design
S9 paired endpoint arrays:   float64
```

NPZ members use canonical lexical order, fixed ZIP metadata, no compression,
no object dtype, no pickle, and finite arrays only. The result writer reloads
all three NPZ files, validates replicate IDs `0..4095` exactly, reconstructs
every aggregate, and requires exact equality before publishing the manifest.
Missing, duplicated, reordered, nonfinite, or object arrays fail closed.

Three shadow artifacts replayed their aggregates exactly after disk reload:

```text
S6 schema shadow: 4,096 rows, exact aggregate replay
S7 schema shadow: 4,096 rows, exact aggregate replay
S9 schema shadow: 8,192 design rows, exact aggregate replay
```

These are schema fixtures, not registered scientific results.

## Authorization V2

The required future schema is:

```text
c85t_direct_pi_authorization_record_v2
```

It must bind:

```text
authorization ID;
exact direct statement;
V2 lock SHA-256;
V2 lock commit;
exact absolute content-addressed output root;
exact external consumption-ledger path;
C85E / active_acquisition / real_data / manuscript = false.
```

The consumption path is self-referential if an ordinary file SHA is used. V2
therefore defines an explicit normalized authorization binding SHA:

```text
1. replace consumption_ledger_path with a fixed schema marker;
2. hash canonical JSON bytes;
3. derive <consumption-root>/<binding-sha>.json;
4. validate the actual record against that derived path.
```

The preflight receipt records both this binding SHA and the ordinary
authorization-file SHA. The binding convention is protocol-bound and tested;
it is not selected at execution time.

Consumption uses `O_WRONLY | O_CREAT | O_EXCL` at the one derived external
path. A pre-existing receipt blocks execution regardless of the proposed
output root. Once created, the authorization remains consumed after success or
failure.

## Private Runtime Capability

The registered capability class is private and has `init=False`. Construction
requires a module-private sentinel. Issuance occurs only after the exclusive
authorization receipt is durably written.

Each capability contains and is checked against its process-issued identity:

```text
authorization binding SHA;
execution-lock SHA;
attempt ID;
exact absolute output root.
```

Registered seed, exact-scenario, Monte Carlo, and proof-candidate functions
require that private object. A string, public constant, arbitrary object,
`None`, or capability with changed attempt/root binding fails. Shadow fixtures
must receive no registered capability.

## C85T And C85V Separation

C85T V2 may freeze:

```text
11 exact/synthetic scenario results;
S6/S7/S9 replicate artifacts;
seven proof candidates;
candidate dispositions;
proof-candidate schema/internal-consistency checks.
```

It may not transition theorem status. Every C85T row has:

```text
historical_status: OPEN
candidate_disposition: one allowed proposal value
formal_status: OPEN
check_class: PROOF_CANDIDATE_SCHEMA_AND_INTERNAL_CONSISTENCY
```

The old same-process token audit is non-operative. The public status-transition
helper refuses all non-OPEN transitions even when supplied a fabricated `PASS`.
Future C85V requires separate PM approval, reads frozen proof candidates,
rederives finite claims independently, and does not rerun Monte Carlo.

Current theorem state:

| Theorem | Formal status | Canonical candidate present |
|---|---|---:|
| T1 | OPEN | 0 |
| T2 | OPEN | 0 |
| T3 | OPEN | 0 |
| T4 | OPEN | 0 |
| T5 | OPEN | 0 |
| T6 | OPEN | 0 |
| T7 | OPEN | 0 |

## Append-Only Lifecycle

The V2 attempt ledger is canonical JSONL. Successful order is:

```text
PREFLIGHT_STARTED
PREFLIGHT_COMPLETED
AUTHORIZATION_CONSUMED
EXACT_SCENARIOS_STARTED
EXACT_SCENARIOS_COMPLETED
MONTE_CARLO_STARTED
MONTE_CARLO_COMPLETED
PROOF_CANDIDATES_STARTED
PROOF_CANDIDATES_COMPLETED
MANIFEST_STARTED
MANIFEST_COMPLETED
ATOMIC_PUBLISH_COMPLETED
```

`FAILED` may terminate at any stage and records the last completed stage,
primary exception type, and primary exception message. Every event records a
strict sequence number, UTC timestamp, authorization SHA, lock SHA, attempt
ID, and artifact/receipt SHA where applicable. Each append uses `O_APPEND` and
`fsync`. Skipped or reordered stages fail.

## V2 Result And Atomic Publication

The result schema is:

```text
c85t_synthetic_validation_and_proof_candidates_result_v2
```

The manifest schema is:

```text
c85t_atomic_result_manifest_v2
```

Mandatory result objects are:

```text
exact_scenario_results.json
monte_carlo_summary.json
S6_replicates.npz
S7_replicates.npz
S9_replicates.npz
S9_raw_draw_digest_registry.csv
seven c85t_proof_candidates/*.md files
proof_candidate_dispositions.csv
authorization_consumed.json
C85T_RESULT.json
C85T_RESULT_ARTIFACT_MANIFEST.json
```

The manifest validates 11 scenario results, 8,192 S6/S7 rows, 8,192 S9
design rows, 4,096 S9 digest rows, seven proof candidates, seven OPEN statuses,
and zero protected counters. It hashes every staged file and publishes by one
rename only after full replay.

Shadow failure injection at `before_result`, `before_manifest`, and
`before_publish` left no final root and preserved a failed staging root.
Automatic retry is false.

## Runtime Replay And Environment

Before future consumption, the coordinator replays:

```text
lock sidecar and schema/status;
five protocol/generator identities;
runtime registry identity;
133 bound repository objects;
path, byte count, SHA-256, and Git blob for every object;
branch oaci;
clean worktree;
HEAD == origin/oaci;
lock and implementation ancestry;
exact Python/NumPy environment files;
fresh authorization and exact output root.
```

Environment:

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

GPU:
  0
```

The pre-existing dual NumPy metadata trees remain disclosed and eleven NumPy
source/binary/metadata files are byte-bound. The lock contains 133 repository
objects totaling 574,330 bytes; its separately bound runtime registry is
21,775 bytes.

## Public Entrypoint

The only operative future command is:

```text
python -m oaci.theory.c85t_execute_v2 run-locked \
  --execution-lock oaci/reports/C85T_EXECUTION_LOCK_V2.json \
  --output-root <EXACT_AUTHORIZED_CONTENT_ADDRESSED_ROOT>
```

The historical `c85t_execute` path is non-operative for proof-status changes.
No notebook, `python -c`, static token, alternate RNG, alternate result writer,
or unbound wrapper can create a valid V2 result.

## Shadow Validation

New C85TR1 test contribution:

```text
execution guard / authorization / lifecycle:
  8

RNG / intervals / replicate persistence / atomicity:
  11

lock / chronology / isolation:
  8

total:
  27
```

All 119 C85P/C85R/C85TL/C85TR1 theory tests passed during component
verification. The lock-specific post-byte-audit run passed 27/27.

## Formal Regression

All accepted formal suites ran on commit
`920c5540a6ae157b77f2acb36f227bfdc172110b`:

| Suite | Result | Runtime | stderr |
|---|---|---:|---:|
| focused C85 | 375 passed | 10.44 s | 0 bytes |
| C65 cumulative | 986 passed, 1 skipped, 3 deselected | 79.89 s | 0 bytes |
| C23 cumulative | 1,397 passed, 1 skipped, 3 deselected | 106.10 s | 0 bytes |
| full OACI | 2,321 passed, 1 skipped, 3 deselected | 315.15 s | 0 bytes |

Every accepted stderr SHA-256 is the empty-file digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The only skip is the historical finalized C78F node. The three standing C79P
unauthorized-adapter nodes are deliberately deselected by the committed
leading-numeric suite wrapper. `squeue` showed zero active C84/C85/OACI jobs;
`sacct` was not used.

## Artifact Inventory

```text
repair protocol/timing/sidecar:          3
new execution-guard/coordinator modules: 2
updated bound implementation modules:    6
readiness/lock builder:                   1
contract tables before runtime registry: 15
runtime registry:                         1
new test files / nodes:                   3 / 27
new regression wrapper:                   1
bound repository objects:                 133
V2 execution locks:                       1
authorization records:                    0
registered result roots:                  0
canonical proof candidates:               0
theorem-status transitions:               0
```

No raw project data, label file, target logit, source artifact, model state,
checkpoint, optimizer state, cache, or file over 50 MiB was added to Git.

## Protected Counters

```text
registered S0-S10 draws:          0
registered scenario results:      0
registered MC replicate rows:     0
canonical proof candidates:       0
independent proof verdicts:        0
theorem-status transitions:        0
C85T authorization records:        0
C85T authorization consumptions:   0
real project data access:          0
active acquisition:                0
training / forward / GPU:          0 / 0 / 0
C85V authorization:                0
C85E authorization:                0
new data/model zoo execution:      0
manuscript work:                    0
```

## Failure Policy

Before authorization consumption, any lock, byte, environment, repository,
authorization, root, or ancestry mismatch stops with no registered draw. After
consumption, the external receipt remains permanent. Any stage failure appends
`FAILED`, preserves the primary exception and failed root, and prohibits
automatic retry. Any implementation-byte change requires a new additive
protocol, lock, and direct authorization.

## Future Boundary

After PM acceptance of this gate, the future shortest direct statement is:

```text
授权 C85T
```

It must be issued after and bound to V2 lock commit `920c5540...` and SHA
`0f6907f9...`. The earlier pre-lock statement does not carry forward.

A successful future C85T V2 must stop at:

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

C85V, C85E, real data, active acquisition, new data/model zoos, and manuscript
work remain unauthorized.

## Disposition

All seven C85TR1 blockers are prospectively repaired and byte-locked. The
historical lock is preserved and non-operative. The V2 lock is technically
ready for a **new** direct PI authorization, while all scientific execution and
proof-status decisions remain unopened.
