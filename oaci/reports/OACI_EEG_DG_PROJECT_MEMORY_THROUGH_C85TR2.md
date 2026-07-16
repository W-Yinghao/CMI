# OACI EEG-DG Project Memory Through C85TR2

## Current State

```text
milestone:
  C85TR2

gate:
  C85T_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_REPAIRED_V3_LOCK_READY_FOR_PI_AUTHORIZATION

V3 lock commit:
  b1a5ba3aca002de7e302fc375298cc69c1ed82a8

V3 lock SHA-256:
  3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9

C85T authorized:
  false

C85V authorized:
  false

C85E authorized:
  false
```

The repository now has one prospective C85T V3 synthetic/proof-candidate
execution path. It has not been authorized or executed.

## Scientific State Preserved

C84S remains the immutable confirmatory result:

```text
primary:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

label frontier:
  C84-L4
```

C84A remains a read-only post-scientific audit. Its COTT average/tail pattern,
MaNo realized action collapse, restricted label-policy interpretation, and
theory bridge do not alter C84-D or C84-L4.

C85P prospectively separates:

```text
statistical experiments;
unrestricted information value;
registered-policy value;
policy approximation gap;
realized policy dependence and collapse;
partial identification and minimax regret;
mean, worst-group, and CVaR target risk;
near-optimal action geometry;
costly full-information label testing.
```

C85R repaired the S0-S10 semantic contract without executing it. The operative
V2 generator SHA-256 remains:

```text
e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a
```

## Historical C85TL And C85TR1 Locks

C85TL fixed execution modes, PCG64DXSM seeding, S6/S7/S9 estimands, the S8
rational LP schema, and proof artifact forms. Its historical lock SHA-256 is:

```text
4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991
```

C85TR1 additively repaired int64 S9 RNG bytes, raw/clipped probability
intervals, replicate persistence, single-use consumption, C85T/C85V stage
separation, and lifecycle evidence. Its V2 lock is:

```text
commit:
  920c5540a6ae157b77f2acb36f227bfdc172110b

SHA-256:
  0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719
```

Both locks are historical and non-operative:

```text
status:
  SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION

authorization records:
  absent

registered execution:
  0
```

Do not use `c85t_execute`, `c85t_execute_v2`, or either old lock to create an
official result.

## C85TR2 Contribution

C85TR2 repairs four pre-authorization V2 blockers:

```text
A. receipt-validated context replaces private-sentinel authority claims;
B. one staging transaction contains results, manifest, lifecycle, and receipt;
C. no fallible required operation occurs after the final rename;
D. semantic replay derives all counts and cross-artifact identities from files.
```

Authoritative additive identities:

```text
protocol commit:
  2e79f304202faffb857610e273ec5510a608080a

protocol SHA-256:
  f9a1db908f34818b7551c0d4f8de65fa7a11e71c41b8e5fe28824f042904a844

implementation commit:
  d489bd428f1c39a6eb399c9697b983d0b143ec80

V3 lock commit:
  b1a5ba3aca002de7e302fc375298cc69c1ed82a8

V3 lock SHA-256:
  3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9

runtime registry SHA-256:
  a4cbdccbd872581e8b2c3ee602850e426d992c25f97591a9a474cce2754e0c55

runtime-bound objects:
  160
```

## Receipt-Validated Execution Context

Future registered execution uses:

```text
ValidatedC85TExecutionContext
```

The only operative factory accepts a committed V3 lock path, a committed V3
authorization-record path, and the exact CLI output root. It internally:

```text
replays lock bytes and Git blobs;
replays branch, clean HEAD, and origin;
replays the committed authorization and chronology;
validates output and consumption-root policies;
atomically creates the external O_EXCL receipt;
fsyncs the receipt and directory;
creates the attempt identity and authorization lifecycle event;
returns a receipt-bound context.
```

Every registered exact, Monte Carlo, proof-candidate, and RNG dispatcher
revalidates the receipt and lifecycle. Fabricated or copied contexts,
cross-attempt/root contexts, and deleted/tampered receipts fail.

This is governance for official result publication, not an adversarial Python
or OS security claim.

## V3 Authorization Contract

Future authorization schema:

```text
c85t_direct_pi_authorization_record_v3
```

It must bind:

```text
direct statement:
  授权 C85T

lock commit and SHA;
one authorization ID;
one exact absolute content-addressed output root;
one external consumption path;
all C85E/active/real-data/manuscript fields false.
```

Consumption is globally single-use through `O_CREAT|O_EXCL`. Receipt and parent
directory are fsynced. A consumed authorization remains consumed after success
or failure.

The output policy is:

```text
parent:
  /projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v3

basename:
  c85t-v3-{lock_sha16}-{authorization_id16}

consumption root:
  /projects/EEG-foundation-model/yinghao/oaci-c85t-authorization-consumption-v3
```

## Atomic Bundle And Recovery

Future C85T creates one staging bundle containing synthetic results, persisted
replicates, proof candidates, manifest, lifecycle, copied consumption receipt,
and completion receipt.

The successful lifecycle is:

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
ATOMIC_PUBLISH_COMMIT_READY
```

All writes, semantic replay, completion evidence, lifecycle events, and fsyncs
finish before publication. Commit performs one `os.replace(staging, final)` and
no required operation afterward.

A valid final bundle after a post-rename crash is recovered as success. If
rename did not occur, no final root is valid. Primary exceptions are never
masked by lifecycle, cleanup, quarantine, recovery, or reporting errors.

## Result-Semantic Replay

V3 requires artifact-derived validation:

```text
exact scenario keys:              S0 through S10
S10 exact values:                 11/40, 0, 3/5, 13/40
S8:                               rational LP certificate
S6 rows:                          4,096
S7 rows:                          4,096
S6/S7 logical rows:               8,192
S9 replicate-design rows:         8,192
S9 raw digest rows:               4,096
S9 raw dtype / counts:            <i8 / 51 / 46
proof candidates:                 T1 through T7
formal theorem statuses:          OPEN / 7
```

S6/S7/S9 selected actions, binary indicators, finite regrets, paired arrays,
and all aggregates are reconstructed from saved NPZ arrays. S9 digest fields
are lowercase 64-hex and all 4,096 registered int64 streams are rerun under the
same consumed context for digest verification.

Proof candidate CSV hashes must match file hashes, and statement hashes must
match lock-bound theorem statements. Result lock/auth/attempt/root/HEAD fields
must match the validated context, and all protected counters must be zero.

## Proof Boundary

At C85TR2 completion:

```text
T1: OPEN
T2: OPEN
T3: OPEN
T4: OPEN
T5: OPEN
T6: OPEN
T7: OPEN

canonical proof candidates:
  0

independent proof verdicts:
  0

status transitions:
  0
```

Future C85T may freeze seven proof candidates and non-dispositive internal
consistency checks. It cannot transition formal status. C85V requires separate
PM approval and must not rerun the Monte Carlo benchmark.

## Future V3 Result Contract

Required future arithmetic:

```text
scenario results:                    11
S6/S7 logical replicate rows:     8,192
S9 logical replicate-design rows: 8,192
S9 raw int64 digest rows:          4,096
proof candidates:                      7
formal theorem statuses OPEN:          7
```

Required future files include:

```text
C85T_RESULT.json
C85T_RESULT_ARTIFACT_MANIFEST.json
C85T_V3_LIFECYCLE.jsonl
C85T_V3_COMPLETION_RECEIPT.json
authorization_consumed.json
exact_scenario_results.json
monte_carlo_summary.json
S6_replicates.npz
S7_replicates.npz
S9_replicates.npz
S9_raw_draw_digest_registry.csv
seven proof-candidate Markdown files
proof_candidate_dispositions.csv
```

## Environment

```text
prefix:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python:
  3.13.7

NumPy runtime / first metadata match:
  2.4.4 / 2.3.3

bit generator:
  PCG64DXSM

CPU / GPU / RAM / wall / storage:
  1 / 0 / 8 GiB / 30 minutes / 64 MiB
```

## Verification

```text
post-lock direct C85TR2:
  35 passed

focused:
  410 passed

C65:
  1,021 passed, 1 skipped, 3 deselected

C23:
  1,432 passed, 1 skipped, 3 deselected

full OACI:
  2,356 passed, 1 skipped, 3 deselected

accepted stderr:
  empty for every suite

final red team:
  70 / 70 PASS
```

The one skip is finalized C78F evidence. The standing deselections are three
historical unauthorized C79 adapter tests.

Two non-accepted readiness invocations are disclosed: a Python 3.9 collection
error and a mistyped initial lock-builder commit. Both failed before registered
execution or authorization consumption.

## Operational Disclosure

`squeue` showed one pre-existing user job:

```text
897842 | bash | /bin/bash | cpu-high | RUNNING
```

C85TR2 did not submit or alter it, and its visible name/command do not identify
C84/C85/OACI work. No `sacct` evidence is claimed.

## Zero Counters

```text
registered draws/results:          0 / 0
registered MC rows:                0
canonical proof candidates:        0
theorem transitions:               0
C85T authorization/consumption:    0 / 0
real project data:                  0
training/forward/GPU:               0 / 0 / 0
active acquisition:                0
C85V/C85E:                          0 / 0
new data/model-zoo execution:       0
manuscript work:                    0
```

## Future Execution Boundary

No current V3 authorization record exists. Authorization text received before
the V3 lock is not reusable. After PM accepts the readiness gate, a new direct
statement is required:

```text
授权 C85T
```

It must bind V3 lock commit `b1a5ba3aca002de7e302fc375298cc69c1ed82a8`
and SHA-256
`3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9`.

Only this command is operative after such authorization:

```text
python -m oaci.theory.c85t_execute_v3 run-locked \
  --execution-lock oaci/reports/C85T_EXECUTION_LOCK_V3.json \
  --authorization-record oaci/reports/C85T_V3_PI_AUTHORIZATION_RECORD.json \
  --output-root <EXACT_AUTHORIZED_CONTENT_ADDRESSED_ROOT>
```

A successful future C85T stops at:

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

It does not authorize C85V, C85E, real data, active acquisition, new data/model
zoos, or manuscript work.

