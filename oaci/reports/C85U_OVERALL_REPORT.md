# C85U Held-Evaluation Candidate-Utility Reconstruction Overall Report

## Final Disposition

```text
C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED
```

C85U consumed one fresh V2 authorization and completed the protected U1 utility
construction, isolated U2 historical-decision replay, and one atomic final
acceptance transaction. The complete 944 x 81 held-evaluation utility field is
now frozen for C85EP2 review.

C85U did not rerun a selector or Q0 chain, recompute scientific inference,
change any C84/C85 result, authorize C85E/C86, execute active acquisition, add
data or models, or modify manuscript prose.

## Authoritative Identities

| Object | Identity |
|---|---|
| C85UR1 readiness HEAD | `be2fad4868407b1aa8f7adf6a59d151efb68e02b` |
| authorization/execution HEAD | `f4b05c3dbed962348efe9cab56374854120a3667` |
| V2 lock commit | `672670d05e9d7adfbe12673d4a64bfd499413162` |
| V2 lock SHA-256 | `77382c16a593f7c2bdeb4dcacdfa21df11dcfd59982e9bfb982d6b88f5f04d1d` |
| authorization file SHA-256 | `024d95b6364651d6faa7b7cbeb5e0a1d896fe56e122d3b4ad2d6ba284ac1b6db` |
| consumption receipt SHA-256 | `f2ae41730a005d5622280ad7617efcd198ab308805604894cddb25d8eb5726b9` |
| protected replay SHA-256 | `9013c5223bf271edcefe477443add3c1404381d5bf9c884b8d283ac3bc94651e` |
| U1 manifest SHA-256 | `95bdbc04f05103a090d46dd4419dc12c766ab45f807c8466ebf883a1171b05c6` |
| U2 result SHA-256 | `84177e80c9883611ef0bc0e9d27a4c38867a45db9b0458d7b090c422b23c39be` |
| final result SHA-256 | `d19b11c24a811c1e8677cc0681d3d57bcb437a1d43702a5df8b2e1c92d43f83c` |
| acceptance manifest SHA-256 | `dfcf84569beb1b34b786cbe72233a22fd3928a4475b7e345f23b40cdb6671620` |
| completion receipt SHA-256 | `5d8bdc9888106f6382531f52150613744cce3e15fbc73e0b557b3eaa89e7a129` |
| lifecycle SHA-256 | `c7ade7f29723fdcaa4f4472e2b431a1ec3c581e07f157f1c4d1d49d60125b2b7` |

External root:

```text
/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/
  c85u-v2-77382c16a593f7c2-91a428488a634268
```

Attempt identities:

```text
authorization ID:
  91a42848-8a63-4268-b865-dc1b6f933a1b

authorization binding SHA-256:
  a866bb820251b603ad29fc9c13b2e1997025678fa9b1a243ff0cf55cc62b1ea7

attempt ID:
  147245c8846d40e5a6059e353fce5b8b
```

## Authorization And Execution

The direct statement `授权 C85U` was bound only to the current V2 lock. The
authorization record was committed and pushed as the sole repository change in
commit `f4b05c3dbed962348efe9cab56374854120a3667`.

Dry preflight established a clean `oaci` branch, `HEAD == origin/oaci`, exact
lock and 54 bound objects, exact environment, authorization chronology, and
absent output/receipt paths. The first shell formulation failed at Python parse
time before lock replay; the corrected dry preflight passed and no
authorization was consumed by either dry check.

The lock-bound coordinator ran as Slurm job 900451:

```text
partition:       cpu-high
requested CPU:   48
memory:          128 GiB
GPU:             0
wall limit:      2 hours
```

The application lifecycle completed in 150.611275 seconds. Slurm stdout records
`classification = SUCCESS`; Slurm and both child-stage stderr files are empty.
The job left `squeue` after completion. A `sacct` query was attempted but its
accounting database connection was unavailable, and `scontrol` no longer
retained the terminal job. No scheduler terminal state or exit code is inferred
from unavailable evidence; success is established by stdout and full frozen
bundle replay.

## Protected Input Replay

After authorization consumption and before U1, the runtime streamed and hashed:

```text
target artifacts:       1,944 / 1,944
target artifact bytes:  48,018,748,054
target sidecars:         1,944 / 1,944
evaluation table rows:   4,848
```

The protected replay receipt binds the same authorization, lock, attempt,
output root, evaluation view, target registry, and sidecar registry.

## Stage U1

U1 opened exactly one immutable evaluation label table and each target artifact
once under its exclusive stage receipt. It produced:

```text
contexts:                     944 / 944
candidates/context:             81 / 81
candidate utility rows:      76,464 / 76,464
context artifacts:              944
U1 output bytes:          44,003,342
```

The 944 context files and 76,464-row index replayed under the historical
bAcc/NLL/ECE, oriented-midrank, first-argmax, and standardized-regret contract.
U1 opened zero selection objects, Q0 shards, direct scientific result tables,
construction labels, target sidecar payloads, or inference paths.

## Stage U2

Only after U1 froze, U2 replayed the immutable Stage-B actions and historical
method-context endpoints:

```text
contexts:                         944 / 944
historical method-context rows: 18,432 / 18,432
finite Q0 action records:    8,749,056 / 8,749,056
selected-regime mismatches:          0
max endpoint replay errors:           0
```

U2 opened zero evaluation labels, target artifacts, target-logit arrays, or
inference paths. It did not resample Q0.

## Atomic Acceptance

The final bundle contains the result, artifact manifest, completion receipt,
lifecycle, copied authorization receipt, protected-replay identity, and U1/U2
identities. All twelve lifecycle events are ordered and exact. The final
acceptance was published by one rename, no staging root remains, and the
production validator replayed the external receipt, all manifest rows, counts,
hashes, lifecycle, completion receipt, and protected counters.

```text
external files:                972
external bytes:         44,028,286
residual staging roots:          0
```

## Regression Verification

| Suite | Job | Accepted result | Pytest time | stderr bytes |
|---|---:|---|---:|---:|
| focused | 900453 | 395 passed, 1 deselected | 10.90 s | 0 |
| C65 | 900454 | 1,087 passed, 1 skipped, 6 deselected | 127.49 s | 0 |
| C23 | 900455 | 1,498 passed, 1 skipped, 6 deselected | 118.87 s | 0 |
| full OACI | 900456 | 2,422 passed, 1 skipped, 6 deselected | 518.44 s | 0 |

The additional deselection is the C85UR1 readiness-only assertion that the V2
authorization must not exist. No implementation, lock, or test was changed.

## Immutable Scientific Boundary

```text
C84 primary:       C84-D
C84 label frontier: C84-L4

T1/T3/T4/T7: PROVED
T2/T6:       COUNTEREXAMPLE
T5:          OPEN

Q0 resampling:             0
selector recomputation:    0
scientific inference:      0
theorem-status writes:     0
C85E / C86:                0 / 0
```

C85U is exploratory infrastructure, not a scientific result or theorem test.
The next permissible milestone is C85EP2, which must independently replay this
field and acceptance bundle before creating any C85E lock. C85E, C86, active
acquisition, new data/model zoos, and manuscript changes remain unauthorized.
