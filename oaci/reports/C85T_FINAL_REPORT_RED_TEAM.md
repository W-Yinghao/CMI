# C85T V3 Final Report Red Team

## Verdict

```text
checks: 80
PASS:   80
FAIL:    0
```

The frozen external bundle supports exactly this gate:

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

This is not an independent proof verdict. It does not transition T1-T7 or
authorize C85V, C85E, real data, active acquisition, or manuscript work.

## Evidence Identity

```text
execution HEAD / authorization commit:
  b26b21f6b8378188dd59890c5701944c41fad823

Slurm job:
  899524

attempt ID:
  3ced6e5b117a4a95b9982a5fd22b5e4a

result SHA-256:
  ecaff65e942dbb81d93a3bdb61589fa9f1f6590f7188947688e6b30617140cec

manifest SHA-256:
  a727beebcb45598ea0f92f37bed8ef32369b1c793ecad9efc2f5d9941bd5bb0e

semantic replay receipt SHA-256:
  735edf13a24c074cb3c18e56d168ebd905b3a7bcb29e3c273b3652bb1b7dcc6e

completion receipt SHA-256:
  418f74e4c3cf60847b11bf18a890ffebf870ed8adee1a75d304b01075646e65d
```

## Audit Matrix

| Category | Checks | Passed |
|---|---:|---:|
| authorization identity and single use | 10 | 10 |
| scheduler and execution identity | 8 | 8 |
| lifecycle and atomic publication | 12 | 12 |
| artifact manifest and byte replay | 12 | 12 |
| result-semantic replay | 16 | 16 |
| proof-candidate governance | 8 | 8 |
| protected scientific boundary | 6 | 6 |
| regressions and operational hygiene | 8 | 8 |
| **Total** | **80** | **80** |

## Authorization Audit

All ten checks pass:

1. The direct statement followed V3 lock creation.
2. The authorization record schema is V3.
3. The authorization ID is new and valid.
4. The record was committed after the lock.
5. `HEAD == origin/oaci` and the worktree was clean before execution.
6. Lock SHA and lock commit match the V3 lock.
7. Output root matches the lock/authorization derivation.
8. Consumption path matches the normalized authorization binding.
9. External receipt was created once and matches the copied receipt.
10. C85V/C85E/active/real-data/new-zoo/manuscript fields are false.

Authorization evidence:

```text
authorization ID:
  9ec012be-dbf2-4f1f-ab99-a5406596c31c

authorization binding SHA-256:
  38b8c3f2111df926c388ba7ab60292aa43714b9f0dace1a2beaa978f30a918fc

authorization file SHA-256:
  b0c283967c7741ebe7eecd0c0207c7dbec7e3f8ccd435db4ae594de41e19e501

consumption receipt SHA-256:
  81651f69513fe5986a47975a5616ede4a2bcab2c82696ab620efc61a9c855d67
```

## Scheduler And Runtime Audit

Job `899524` was submitted with one task, one requested CPU, 8 GiB memory,
30-minute wall limit, and zero GPU. Slurm allocated two CPU threads while
retaining `CPUs/Task=1`; this does not change the serial locked runtime.

Observed through `squeue` during execution and `scontrol` after completion:

```text
state:          COMPLETED
exit code:      0:0
runtime:        00:00:07
partition:      cpu-high
node:           nodecpu01
GPU:            0
stderr bytes:   0
```

No `sacct` evidence is claimed.

## Lifecycle And Atomic Publication

All twelve required lifecycle events exist exactly once and in order. The
terminal event is `ATOMIC_PUBLISH_COMMIT_READY`. The final bundle exists and
the attempt staging directory is absent.

The completion receipt binds the result, manifest, lifecycle prefix, semantic
receipt, external consumption receipt, lock, authorization, attempt, root, and
execution HEAD. The read-only recovery validator classifies the existing final
bundle as a valid post-rename success. This validation does not imply that an
actual crash occurred.

## Artifact Manifest Audit

The bundle contains 21 files totaling 1,325,040 bytes. The artifact manifest
covers 18 payload artifacts; the lifecycle, manifest itself, and completion
receipt are the three transaction control files.

All manifest rows replay path, size, and SHA-256. No missing or extra file was
found. Important identities:

```text
exact results:
  dde13367e7c43403d54e196a250dc7c6b5692f7fc0323b1ceb015d0aa0e39d0f

Monte Carlo summary:
  60c60f4f020710e7e825330e30ea31c218cfa66b154a7b1b44a35a4b72c6b594

S6 NPZ:
  6e967deff1f0bcda82ae7424256326fa4da35fba7c9c18629c0c8d7d57742321

S7 NPZ:
  bc62b5c73956ff094241d72f5756eb80b66cd23f4b61272a10382540061e9781

S9 NPZ:
  9e7e1d29ab8acab69f2e753465e69481fe139af877d92321f45b9222ef66b9c3

S9 digest registry:
  7211ed0ac6a4f43ab76a0604bcaab35bdc65218fc83eb8ad8d98ad1df279bb5d
```

## Semantic Replay Audit

All sixteen checks pass:

```text
scenario keys:                    S0-S10 exactly
scenario count:                   11
S6 rows:                          4,096
S7 rows:                          4,096
S6/S7 rows:                       8,192
S9 design rows:                   8,192
S9 digest rows:                   4,096
registered RNG digest replays:    4,096
proof candidates:                 7
formal OPEN statuses:             7
protected counters zero:          true
```

S10 replays `11/40`, `0`, `3/5`, and `13/40`. S8 contains the rational LP
certificate. S6/S7/S9 aggregates are derived from saved arrays. All S9 int64
digests were checked under the consumed attempt. The semantic receipt says
`SEMANTIC_REPLAY_PASS`.

## Proof-Candidate Governance

Exactly T1-T7 are frozen. Candidate dispositions are:

```text
T1  PROPOSED_PROOF
T2  PROPOSED_COUNTEREXAMPLE
T3  PROPOSED_PROOF
T4  PROPOSED_PROOF
T5  INCOMPLETE_OPEN
T6  PROPOSED_COUNTEREXAMPLE
T7  PROPOSED_PROOF
```

Every candidate has the required statement hash, assumptions, boundary cases,
and internal-consistency label. These checks are explicitly non-dispositive.
Every formal theorem status remains `OPEN`.

## Protected Boundary Audit

```text
real project data access:  0
active acquisition:        0
training / forward / GPU:  0 / 0 / 0
theorem transitions:       0
C85V / C85E authorized:    false / false
manuscript modified:        false
```

No empirical OACI field, EEG array, label view, logits, checkpoints, or model
state entered C85T.

## Regression Audit

```text
focused: 409 passed, 1 deselected
C65:     1,020 passed, 1 skipped, 4 deselected
C23:     1,431 passed, 1 skipped, 4 deselected
full:    2,355 passed, 1 skipped, 4 deselected
```

All accepted stderr files are empty. The additional post-execution deselection
is the readiness-only assertion that an authorization record must be absent.
No lock-bound test was edited.

Two non-accepted operational invocations are disclosed: one wrong-Python
read-only collector import and one stale-SLURM-ID allocation rejection. Neither
ran a registered stream or altered the bundle.

## Active Job Disclosure

At final reporting, `squeue` showed jobs `897842` (`bash`) and `899527`
(`targetx-full`). Neither was submitted or modified by C85T. No active job with
the C85T name or V3 entrypoint remained.

## Final Boundary

The final gate is accepted exactly as frozen:

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

C85V remains unapproved. C85T results cannot themselves transition theorem
status or authorize any real-data, acquisition, model, dataset, or manuscript
work.

