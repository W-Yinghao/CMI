# C85TR2 Overall Report

## Final Disposition

```text
C85T_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_REPAIRED_V3_LOCK_READY_FOR_PI_AUTHORIZATION
```

C85TR2 completed an additive execution-governance repair and created one
prospective C85T V3 execution lock. It repaired receipt validation, atomic
publication, failure recovery, and result-semantic replay without opening any
registered S0-S10 stream or producing a canonical proof candidate.

The V3 lock is ready for PM review and a future fresh direct authorization. It
is not currently authorized.

## Scope And Non-Scope

C85TR2 performed only:

```text
protocol chronology and historical-lock preservation;
authorization-certificate and receipt validation implementation;
single-transaction publication and recovery implementation;
primary-exception preservation;
artifact-derived V3 semantic replay;
shadow/adversarial tests;
execution-lock creation;
focused and cumulative regressions;
readiness, red-team, memory, and handoff reporting.
```

C85TR2 did not perform:

```text
registered S0-S10 execution;
registered Monte Carlo draws;
canonical proof-candidate generation;
theorem-status transition;
C85T authorization or consumption;
C85V review;
real project data access;
active acquisition;
training, forward, or GPU work;
C85E work;
new data/model-zoo execution;
manuscript work.
```

## Authoritative Identities

| Object | Commit / SHA-256 |
|---|---|
| entering accepted HEAD | `dd75d52be4414cc893c5a2fddf0374e01e13137a` |
| C85TR2 protocol commit | `2e79f304202faffb857610e273ec5510a608080a` |
| C85TR2 protocol SHA-256 | `f9a1db908f34818b7551c0d4f8de65fa7a11e71c41b8e5fe28824f042904a844` |
| C85TR2 implementation commit | `d489bd428f1c39a6eb399c9697b983d0b143ec80` |
| C85T V3 lock commit | `b1a5ba3aca002de7e302fc375298cc69c1ed82a8` |
| C85T V3 lock SHA-256 | `3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9` |
| V3 runtime registry SHA-256 | `a4cbdccbd872581e8b2c3ee602850e426d992c25f97591a9a474cce2754e0c55` |
| historical V2 lock SHA-256 | `0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719` |

The V3 lock binds 160 repository objects totaling 792,406 bytes, plus the
runtime registry's own 26,163 bytes and Git blob identity.

## Additive Chronology

The protocol and timing audit were committed and pushed first. Only then were
the V3 implementation and shadow contracts written. The lock was generated
after the implementation commit and binds that exact commit.

```text
2e79f304  C85TR2 protocol
    ->
d489bd42  implementation and shadow/adversarial validation
    ->
b1a5ba3a  V3 runtime registry and execution lock
```

At every chronology boundary:

```text
registered S0-S10 draws:      0
canonical proof candidates:   0
theorem-status transitions:   0
C85T authorization records:   0
```

## Historical V2 Preservation

The V2 lock is preserved unchanged. Its supersession record states:

```text
status:
  SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION

authorization record:
  absent

authorization consumption:
  0

registered execution:
  0
```

No V2 artifact was rewritten as if the C85TR2 fixes had always existed. The V2
lock remains historical evidence and cannot be used as an operative fallback.

## Blocker A Repair: Validated Execution Context

The V2 private-sentinel design was a Python convention, not a robust official
result boundary. V3 removes that claim and uses:

```text
ValidatedC85TExecutionContext
```

The only operative factory requires:

```text
committed V3 lock path;
committed V3 authorization-record path;
exact CLI output root.
```

The factory internally and inseparably performs:

1. V3 lock sidecar, schema, status, protocol, file-byte, and Git-blob replay.
2. Branch `oaci`, clean worktree, `HEAD == origin/oaci`, and ancestry replay.
3. Committed authorization path, schema, chronology, statement, lock, output,
   and protected-field replay.
4. Exact output basename/parent and consumption-root policy replay.
5. Atomic external receipt creation with `O_CREAT|O_EXCL`.
6. Receipt-file and containing-directory `fsync`.
7. Attempt-ID and matching `AUTHORIZATION_CONSUMED` lifecycle creation.
8. Construction of a context bound to the receipt inode, ordinary
   authorization SHA, normalized authorization binding SHA, authorization ID,
   lock SHA/commit, attempt, output root, and HEAD.

Every registered exact scenario, Monte Carlo, proof-candidate, and registered
RNG dispatcher revalidates the context, external receipt, and lifecycle
identity. The implementation rejects:

```text
fabricated dataclasses;
copied contexts;
contexts from another attempt or root;
deleted or byte-tampered receipts;
arbitrary pre-parsed authorization mappings;
authorization reuse after an existing receipt.
```

This is explicitly an official-result governance mechanism. It is not claimed
to provide adversarial code-execution security against a user who can modify
the process or filesystem.

## Blocker B Repair: One Atomic Execution Bundle

V3 creates one bundle with schema:

```text
c85t_atomic_execution_bundle_v3
```

The staging bundle contains all synthetic results, replicate artifacts, proof
candidates, manifest, lifecycle, copied consumption receipt, and completion
receipt. Its successful lifecycle is:

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

Before publication, V3:

```text
writes every required file;
replays all artifact semantics;
writes and replays the manifest;
writes the completion receipt;
writes the terminal lifecycle event;
fsyncs every file and directory;
validates the complete staging bundle.
```

The final commit path then performs exactly one required filesystem operation:

```text
os.replace(staging_bundle, final_bundle)
```

It returns only precomputed in-memory completion data. There is no callback,
lifecycle append, manifest replay, completion write, or required reporting
operation after rename.

## Blocker C Repair: Terminal Recovery And Exception Precedence

A valid terminal final bundle is authoritative even if the process crashes
after rename but before returning to the shell. Recovery classifies it as:

```text
SUCCESS_RECOVERED_AFTER_RENAME
```

If rename did not occur, no final root is valid. Staging is retained or
quarantined as failure evidence.

Failure handling follows these precedence rules:

```text
nonterminal lifecycle:
  append FAILED once

terminal lifecycle + valid final bundle:
  recover success

terminal lifecycle + no final bundle:
  emit reconciliation blocker separately

secondary lifecycle/cleanup/quarantine/reporting errors:
  retain in a secondary-error list

primary exception:
  never replace or mask
```

The shadow tests inject lifecycle append failure and terminal-staging failure
and verify that the original exception remains the primary one.

## Blocker D Repair: Artifact-Derived Semantic Replay

The V3 validator derives counts and identities from persisted artifacts rather
than returning hard-coded pass counts.

### Exact scenarios

It requires exact keys `S0` through `S10`, no missing or extra key, and
scenario-specific fields. It replays:

```text
S10 coarse risk:                 11/40
S10 rich unrestricted risk:      0
S10 rich registered risk:        3/5
S10 reversal:                    13/40
```

S8 must contain the complete rational LP certificate, including identified-set
diameter, randomized solution, minimax value, extreme-point slacks, pure-action
value, and randomization gain.

### S6 and S7

Each NPZ must contain exactly 4,096 ordered unique replicate IDs. Selected
actions must fall within the scenario action range; indicators are binary;
regrets are finite and nonnegative. All probability, regret, interval, and
effective-geometry aggregates are reconstructed from the saved arrays.

### S9

V3 requires:

```text
replicate IDs per design:       4,096
logical design rows:            8,192
selected action domain:         {0,1,2,3}
raw digest rows:                4,096
canonical raw dtype:            <i8
L/H counts:                     51 / 46
SHA format:                     lowercase 64-hex
```

The validator checks binary indicators, nonnegative regrets, paired endpoint
arrays, allocation identities, and exact aggregate reconstruction. Because raw
draws remain digest-only, it reruns the registered PCG64DXSM int64 stream under
the already consumed context and verifies all L, H, and combined digests. This
is replay within the same authorized attempt and does not add scientific
replicates.

### Proof candidates

Proof artifacts must cover exactly T1-T7. Every theorem has one Markdown file
and one disposition row. The CSV candidate SHA must equal the file SHA, and the
statement SHA embedded in the file must equal the lock-bound theorem statement.
Only non-dispositive candidate dispositions are accepted, and every formal
status remains `OPEN`.

### Authorization and result identity

The copied consumption receipt must exactly match the external receipt. Result
lock, authorization, attempt, root, and HEAD must match the validated context.
All real-data, active-acquisition, training, forward, GPU, C85V/C85E, and
manuscript counters must be zero.

## C85T And C85V Separation

C85T V3 may execute the locked synthetic benchmark and freeze seven proof
candidates after a future fresh authorization. It may not claim independent
proof review or transition theorem status.

```text
T1: OPEN
T2: OPEN
T3: OPEN
T4: OPEN
T5: OPEN
T6: OPEN
T7: OPEN
```

C85V remains a separate, read-only, PM-approved future milestone. C85TR2 does
not authorize it.

## V3 Lock Contract

```text
schema:
  c85t_execution_lock_v3

status:
  LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED

authorization schema:
  c85t_direct_pi_authorization_record_v3

authorization record path:
  oaci/reports/C85T_V3_PI_AUTHORIZATION_RECORD.json

output parent:
  /projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v3

output basename:
  c85t-v3-{lock_sha16}-{authorization_id16}

consumption root:
  /projects/EEG-foundation-model/yinghao/oaci-c85t-authorization-consumption-v3

CPU / GPU / RAM / wall / storage:
  1 / 0 / 8 GiB / 30 minutes / 64 MiB
```

The only operative future entrypoint is:

```text
python -m oaci.theory.c85t_execute_v3 run-locked \
  --execution-lock oaci/reports/C85T_EXECUTION_LOCK_V3.json \
  --authorization-record oaci/reports/C85T_V3_PI_AUTHORIZATION_RECORD.json \
  --output-root <EXACT_AUTHORIZED_CONTENT_ADDRESSED_ROOT>
```

No unbound notebook, `python -c`, alternate wrapper, or V1/V2 coordinator can
publish an official V3 result.

## Result Bundle Contract

A future successful V3 bundle must contain:

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

Required arithmetic:

```text
scenario results:                    11
S6/S7 logical replicate rows:     8,192
S9 logical replicate-design rows: 8,192
S9 raw digest rows:               4,096
proof candidates:                     7
formal OPEN statuses:                 7
```

Future success stops at:

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

## Validation Evidence

### Shadow and lock tests

```text
pre-lock direct C85TR2 tests:
  26 passed

post-lock direct C85TR2 tests:
  35 passed

historical C85TR1 compatibility tests:
  27 passed
```

These tests use shadow fixtures only. They do not open a registered scenario
stream or render a canonical proof candidate.

### Formal regressions

| Suite | Result | Pytest time | Wall | Stderr |
|---|---:|---:|---:|---:|
| focused | 410 passed | 12.62 s | 13 s | 0 bytes |
| C65 | 1,021 passed, 1 skipped, 3 deselected | 81.39 s | 84 s | 0 bytes |
| C23 | 1,432 passed, 1 skipped, 3 deselected | 113.61 s | 116 s | 0 bytes |
| full OACI | 2,356 passed, 1 skipped, 3 deselected | 322.22 s | 325 s | 0 bytes |

All accepted stderr hashes equal the SHA-256 of the empty byte string:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

External accepted logs are retained at:

```text
/home/infres/yinwang/CMI_AAAI/c85tr2_regression_logs/
```

The one skip is the finalized historical C78F test. The three standing
deselections are historical unauthorized C79 adapter tests.

## Disclosed Non-Accepted Readiness Attempts

Two errors occurred before accepted validation:

1. An initial test invocation used login Python 3.9 and failed during
   collection because historical code requires Python 3.13
   `dataclass(slots=True)`. No test or registered path ran. Accepted runs used
   the exact locked Python 3.13 environment.
2. An initial lock-builder command passed a mistyped expanded implementation
   commit and failed before writing a registry or lock. The accepted builder
   used exact implementation commit
   `d489bd428f1c39a6eb399c9697b983d0b143ec80`.

Neither event consumed authorization, opened a registered seed, created a
proof candidate, or changed theorem status.

## Red-Team Result

```text
checks: 70
PASS:   70
FAIL:    0
```

The independent report is
`oaci/reports/C85TR2_FINAL_REPORT_RED_TEAM.md`.

## Environment And Git Hygiene

```text
environment:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python:
  3.13.7

NumPy runtime / first metadata match:
  2.4.4 / 2.3.3

bit generator:
  PCG64DXSM

GPU:
  0

largest tracked file:
  21,936,073 bytes

Git file envelope:
  < 50 MiB PASS

raw EEG / checkpoints / model weights added:
  0 / 0 / 0
```

At the final operational audit, `squeue` showed one pre-existing user job:

```text
897842 | bash | /bin/bash | cpu-high | RUNNING
```

C85TR2 did not submit or modify this job. Its visible name and command do not
identify C84/C85/OACI work. No `sacct` evidence is claimed.

## Protected Counters

```text
registered S0-S10 draws:          0
registered scenario results:      0
registered MC replicate rows:     0
canonical proof candidates:       0
independent proof verdicts:        0
theorem-status transitions:       0
C85T authorization records:       0
C85T authorization consumptions:  0
real project data access:          0
training / forward / GPU:          0 / 0 / 0
active acquisition:                0
C85V / C85E authorization:         0 / 0
new data/model-zoo execution:      0
manuscript work:                   0
```

## Authorization Interpretation

The `授权 C85T` text supplied before the V3 lock existed is not reusable. No V3
authorization record was created, committed, or consumed.

After PM accepts C85TR2, only a new standalone:

```text
授权 C85T
```

may be bound to:

```text
lock commit:
  b1a5ba3aca002de7e302fc375298cc69c1ed82a8

lock SHA-256:
  3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9

one committed V3 authorization record;
one generated authorization ID;
one exact content-addressed output root;
one external single-use consumption receipt.
```

C85V, C85E, active acquisition, real data, new datasets/model zoos, and
manuscript work remain unauthorized.

## Final Gate

No unresolved C85TR2 blocker remains. The repository therefore stops at:

```text
C85T_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_REPAIRED_V3_LOCK_READY_FOR_PI_AUTHORIZATION
```

