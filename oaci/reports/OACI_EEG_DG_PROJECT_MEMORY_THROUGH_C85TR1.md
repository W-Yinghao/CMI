# OACI EEG-DG Project Memory Through C85TR1

## Current State

```text
milestone:
  C85TR1

gate:
  C85T_EXECUTION_GUARD_RNG_REPLICATE_PERSISTENCE_AND_PROOF_REVIEW_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION

V2 lock commit:
  920c5540a6ae157b77f2acb36f227bfdc172110b

V2 lock SHA-256:
  0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719

C85T authorized:
  false

C85V authorized:
  false

C85E authorized:
  false
```

The repository has one prospective C85T V2 proof-candidate/synthetic execution
path. It has not been authorized or executed.

## Scientific State Preserved

C84S remains the confirmatory multi-dataset result:

```text
primary:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

label frontier:
  C84-L4
```

C84A remains a read-only exploratory audit. Its COTT average/tail, MaNo action
collapse, restricted label-policy, and theory-gap descriptions do not alter
C84-D or C84-L4.

C85P prospectively separates:

```text
statistical experiment;
unrestricted information value;
registered-policy value;
policy approximation gap;
realized action dependence and collapse;
partial identification and minimax regret;
mean, worst-group, and CVaR target risk;
near-optimal action geometry;
costly full-information label testing.
```

C85R repaired the scenario semantics without execution. The operative V2
generator identity remains:

```text
e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a
```

## C85TL Historical State

C85TL fixed exact versus Monte Carlo scenario modes, PCG64DXSM seeds, S6/S7
and S9 estimands, S8 rational LP outputs, proof artifact schemas, and one
coordinator. Its historical lock is preserved:

```text
commit:
  9d414ebb889b2cfc3fefa19fa98d7ea5ca9fd691

SHA-256:
  4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991

authorization:
  absent

registered execution:
  0

status:
  SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION
```

Do not use this historical lock or `c85t_execute` as an operative path.

## C85TR1 Contribution

C85TR1 repairs seven execution-governance blockers:

```text
1. S9 raw Rademacher draw dtype follows protocol int64;
2. probability intervals preserve raw and clipped forms;
3. S6/S7/S9 replicate arrays are mandatory and replayed;
4. authorization is globally single-use and root-bound;
5. registered execution uses a private process-issued capability;
6. C85T freezes proof candidates while C85V reviews them independently;
7. lifecycle evidence is ordered append-only JSONL.
```

The additive repair identities are:

```text
protocol commit:
  46442b281d61d00a575fae17685648b749659263

protocol SHA-256:
  9c0a7084a7ddd83ef96b8d7f95faf89138829729c0acc5c3d6baeb0ef87ab13d

implementation commit:
  f17e25d0d8dc117f7973f90743e07139eeb0c1e1

lock commit:
  920c5540a6ae157b77f2acb36f227bfdc172110b

lock SHA-256:
  0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719
```

## S9 RNG Contract

Future registered S9 execution uses:

```text
Generator:
  numpy.random.Generator(numpy.random.PCG64DXSM(seed))

seed:
  low64(SHA256(C85_SYNTHETIC_V1|S9|replicate_id))

replicates:
  0..4095

draw calls:
  51 int64 values for L, then 46 int64 values for H

mapping:
  0 -> -1, 1 -> +1

canonical digest dtype:
  little-endian int64

passive prefixes:
  51 L / 13 H

Neyman prefixes:
  18 L / 46 H
```

The historical uint8 path is shadow-only. It must never be substituted for the
V2 registered bytes.

## Replicate And Interval Evidence

Future result requirements:

```text
S6_replicates.npz: 4,096 logical rows
S7_replicates.npz: 4,096 logical rows
S9_replicates.npz: 8,192 replicate-design rows
S9 raw digest CSV: 4,096 rows
```

Every aggregate is recomputed from reloaded deterministic NPZ arrays before
publication. Object, nonfinite, missing, duplicate, or reordered data fails.

Probability intervals record both the raw Wald interval and its [0,1]-clipped
reported form. Regret intervals are not clipped.

## Authorization And Capability

Future authorization schema:

```text
c85t_direct_pi_authorization_record_v2
```

It binds a unique authorization ID, direct statement, lock SHA/commit, exact
content-addressed root, and external single-use ledger path. The binding SHA
uses a locked normalization for the self-referential ledger path; the ordinary
record file SHA is separately persisted.

Consumption uses `O_CREAT|O_EXCL`. A consumed record cannot be reused after
success or failure and cannot migrate to another root.

Registered execution requires a private capability issued only after the
exclusive receipt. The capability binds authorization SHA, lock SHA, attempt
ID, and root. No public token or string is sufficient.

## Proof Boundary

At C85TR1 completion:

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

Future C85T can freeze seven proof candidates and internal schema checks, but
cannot transition a status. Future C85V requires separate PM approval and is
the only stage that may issue an independent proof verdict. C85V must not rerun
the Monte Carlo benchmark.

## Lifecycle And Atomic Result

Future successful lifecycle order:

```text
preflight;
authorization consumption;
exact scenarios;
Monte Carlo;
proof candidates;
manifest;
atomic publish.
```

Events are canonical JSONL, append-only, ordered, bound to authorization/lock/
attempt, and `fsync`-backed. A failure is terminal, preserves its primary
exception and last completed stage, and cannot trigger automatic retry.

The V2 result manifest requires 11 scenario results, exact replicate row
counts, 4,096 S9 draw digests, seven proof candidates, seven OPEN statuses,
and zero real-data/active counters before one atomic rename.

## Environment And Lock Replay

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

The dual NumPy metadata state and eleven NumPy files are byte-bound. The V2
lock binds 133 repository objects plus its runtime registry. Future runtime
requires branch `oaci`, a clean tree, `HEAD == origin/oaci`, and exact
SHA-256/Git blobs.

## Verification

```text
new C85TR1 nodes:
  27

focused:
  375 passed

C65:
  986 passed, 1 skipped, 3 deselected

C23:
  1,397 passed, 1 skipped, 3 deselected

full:
  2,321 passed, 1 skipped, 3 deselected

accepted stderr:
  empty

final red team:
  96 / 96 PASS

active C84/C85/OACI jobs via squeue:
  0

sacct used:
  false
```

## Authorization Interpretation

The `授权 C85T` string supplied in the C85TR1 request arrived before V2 lock
creation. It did not create a V2 record and does not carry forward.

After PM accepts the readiness gate, a **new** direct statement is required:

```text
授权 C85T
```

It must bind lock commit `920c5540...` and SHA `0f6907f9...` plus one generated
authorization ID and one exact content-addressed output root.

## Future Execution

Only the following entrypoint is operative:

```text
python -m oaci.theory.c85t_execute_v2 run-locked \
  --execution-lock oaci/reports/C85T_EXECUTION_LOCK_V2.json \
  --output-root <EXACT_AUTHORIZED_CONTENT_ADDRESSED_ROOT>
```

A successful future C85T stops at:

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

It does not authorize C85V, C85E, real data, active acquisition, new data/model
zoos, or manuscript work.

## Zero Counters

```text
registered draws/results:          0 / 0
canonical proof candidates:        0
independent proof verdicts:         0
theorem transitions:               0
C85T authorization/consumption:    0 / 0
real data:                          0
training/forward/GPU:               0 / 0 / 0
active acquisition:                0
C85V/C85E:                          0 / 0
manuscript work:                    0
```
