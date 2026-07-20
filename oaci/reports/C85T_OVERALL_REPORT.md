# C85T V3 Overall Report

## Final Disposition

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

C85T V3 successfully consumed one fresh authorization, executed the exact
locked synthetic benchmark, persisted all registered replicate arrays and S9
raw-draw digests, froze seven proof candidates, and atomically published one
complete execution bundle.

All formal theorem statuses remain `OPEN`. C85T performed no real-data access,
active acquisition, training, forward pass, GPU work, or manuscript action.

## Authoritative Identities

| Object | Identity |
|---|---|
| readiness HEAD | `3f0bd59142f0d6b1807f267851fe49704a410601` |
| authorization/execution HEAD | `b26b21f6b8378188dd59890c5701944c41fad823` |
| V3 lock commit | `b1a5ba3aca002de7e302fc375298cc69c1ed82a8` |
| V3 lock SHA-256 | `3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9` |
| authorization binding SHA-256 | `38b8c3f2111df926c388ba7ab60292aa43714b9f0dace1a2beaa978f30a918fc` |
| authorization file SHA-256 | `b0c283967c7741ebe7eecd0c0207c7dbec7e3f8ccd435db4ae594de41e19e501` |
| external consumption receipt SHA-256 | `81651f69513fe5986a47975a5616ede4a2bcab2c82696ab620efc61a9c855d67` |
| result SHA-256 | `ecaff65e942dbb81d93a3bdb61589fa9f1f6590f7188947688e6b30617140cec` |
| result manifest SHA-256 | `a727beebcb45598ea0f92f37bed8ef32369b1c793ecad9efc2f5d9941bd5bb0e` |
| semantic replay SHA-256 | `735edf13a24c074cb3c18e56d168ebd905b3a7bcb29e3c273b3652bb1b7dcc6e` |
| completion receipt SHA-256 | `418f74e4c3cf60847b11bf18a890ffebf870ed8adee1a75d304b01075646e65d` |

External result root:

```text
/projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v3/
  c85t-v3-3ee51a994969ebaa-9ec012bedbf24f1f
```

Attempt identity:

```text
authorization ID:
  9ec012be-dbf2-4f1f-ab99-a5406596c31c

attempt ID:
  3ced6e5b117a4a95b9982a5fd22b5e4a
```

## Authorization And Preflight

The PI issued the required new standalone direct statement after the V3 lock.
The authorization record was created from the lock-bound derivation API,
committed, and pushed as the only repository change in commit
`b26b21f6b8378188dd59890c5701944c41fad823`.

Before submission:

```text
branch:                         oaci
HEAD == origin/oaci:            true
worktree clean:                 true
lock and 160 bound objects:     PASS
authorization after lock:       PASS
output root absent:             true
staging root absent:            true
consumption receipt absent:     true
active C85T attempts:           0
```

The V3 context created one external `O_CREAT|O_EXCL` receipt and fsynced it and
its containing directory before any registered seed was opened. The receipt
remains consumed permanently.

Protected authorization fields:

```text
C85V:                       false
C85E:                       false
active acquisition:         false
real data:                  false
new data or model zoo:      false
manuscript:                 false
```

## Scheduler Execution

The exact lock-bound entrypoint ran as Slurm job `899524`:

```text
job name:            C85T-V3
partition:           cpu-high
requested CPU:       1
allocated CPU:       2
tasks / CPUs-task:   1 / 1
memory:              8 GiB
GPU:                 0
wall limit:          30 minutes
node:                nodecpu01
state:               COMPLETED
exit code:           0:0
scheduler runtime:   7 seconds
```

The scheduler allocated two CPU threads for a one-CPU request while preserving
one task and `CPUs/Task=1`. The runtime remained serial and the locked
scientific execution was unchanged.

Job stdout contains only the final completion object. Job stderr is empty:

```text
stdout SHA-256:
  dbecc8961ab0793586c703b5b767e519dbcbba84b8d738f8409992772cec82ea

stderr bytes:
  0

stderr SHA-256:
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Execution was monitored with `squeue`. `scontrol` supplied the retained final
job record. No `sacct` evidence is claimed.

## Lifecycle And Atomic Publication

The lifecycle contains exactly these 12 events:

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

The events span `2026-07-17T02:05:34Z` through
`2026-07-17T02:05:38Z`. All result writes, semantic replay, manifest replay,
completion receipt creation, terminal lifecycle write, and fsync completed
before the single final rename.

After publication:

```text
final bundle exists:       true
staging bundle exists:     false
terminal lifecycle:        true
completion receipt:        valid
manifest chain:            valid
external/copy receipt:     byte-identical
```

The read-only recovery validator also accepts the final bundle as a valid
post-rename success. That check validates recovery semantics; it does not mean
the successful process crashed.

## Bundle Inventory

```text
total files:
  21

manifest payload artifacts:
  18

total external bytes:
  1,325,040
```

The three non-payload transaction controls are the manifest, lifecycle, and
completion receipt. All 18 payload artifacts replay exact path, size, and
SHA-256. There are no missing or extra files.

Major persisted artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `C85T_RESULT.json` | 3,144 | `ecaff65e942dbb81d93a3bdb61589fa9f1f6590f7188947688e6b30617140cec` |
| `exact_scenario_results.json` | 1,861 | `dde13367e7c43403d54e196a250dc7c6b5692f7fc0323b1ceb015d0aa0e39d0f` |
| `monte_carlo_summary.json` | 4,559 | `60c60f4f020710e7e825330e30ea31c218cfa66b154a7b1b44a35a4b72c6b594` |
| `S6_replicates.npz` | 58,554 | `6e967deff1f0bcda82ae7424256326fa4da35fba7c9c18629c0c8d7d57742321` |
| `S7_replicates.npz` | 58,554 | `bc62b5c73956ff094241d72f5756eb80b66cd23f4b61272a10382540061e9781` |
| `S9_replicates.npz` | 307,306 | `9e7e1d29ab8acab69f2e753465e69481fe139af877d92321f45b9222ef66b9c3` |
| `S9_raw_draw_digest_registry.csv` | 863,216 | `7211ed0ac6a4f43ab76a0604bcaab35bdc65218fc83eb8ad8d98ad1df279bb5d` |

## Exact Scenario Results

The following are frozen synthetic results. They are not empirical EEG
findings and do not alter C84 science.

### S0-S2: Information And Policy Collapse

```text
S0 optimal constant risk:
  1/2

S1 coarse registered risk:
  1/2

S1 rich registered risk:
  1

S1 rich unrestricted risk:
  0

S2 action divergence:
  0

S2 reference / registered risk:
  7/25 / 7/25
```

S1 is the locked restricted-policy nonmonotonicity example. S2 is the locked
policy-collapse example. Neither is a theorem-status transition.

### S3-S4: Rank, Top-k, And Regret Separation

```text
S3 selected action:
  1

S3 regret:
  0.0050000000000000044

S3 Spearman:
  -0.2571428571428571

S4 top-4 localization:
  1

S4 selected regret:
  4/5
```

These examples establish the synthetic distinction between global ranking,
top-k localization, and selected-action regret under their fixed finite laws.

### S5: Mean/Tail CVaR Proof Target

```text
candidate alpha region:
  (13/20, 1)

endpoint policy:
  both endpoints excluded

status:
  CANDIDATE_PROOF_TARGET_NOT_PROVED_BY_EXECUTION_MODE
```

C85T does not mark T6 proved. The exact candidate region remains for C85V
review of the proof candidate.

### S6-S7: Near-Optimal Geometry

```text
S6 near-optimal count:
  5

S6 Hill-2 / entropy effective size:
  4.902454545613457 / 4.950568827806065

S6 T7 primary bound:
  1.3863432936411313e-49

S7 near-optimal count:
  1

S7 Hill-2 / entropy effective size:
  1.000000000027776 / 1.0000000003610865

S7 T7 primary bound:
  1.1769109439216723e-34
```

### S8: Partial-Identification Minimax Regret

```text
identified-set infinity diameter:
  1

optimal randomized action distribution:
  (8/21, 1/3, 2/7)

minimax regret:
  44/105

pure-action minimax regret:
  4/5

randomization gain:
  8/21

constraint slacks:
  0 / 0 / 0
```

### S9: Full-Information Costly-Label Design

```text
passive allocation:
  51 / 13

Neyman allocation:
  18 / 46

passive analytic variance:
  1327/10359375

Neyman analytic variance:
  317/6468750

population mean losses:
  3/10, 7/20, 13/20, 17/20
```

### S10: Information Dominance And Policy Approximation Reversal

```text
coarse risk:
  11/40

rich unrestricted risk:
  0

rich registered risk:
  3/5

registered-policy reversal:
  13/40
```

## Monte Carlo Results

All Monte Carlo summaries were reconstructed from the persisted arrays after
reload. The 4,096 S9 raw int64 digest rows were deterministically replayed
under the same consumed attempt.

### S6

```text
replicates:
  4,096

top-1 probability:
  0.25048828125

outside-A-epsilon probability:
  0

mean regret:
  0.0017648925781250015

selected-action counts 0..4:
  1026 / 894 / 830 / 709 / 637
```

### S7

```text
replicates:
  4,096

top-1 probability:
  1

outside-A-epsilon probability:
  0

mean regret:
  0

selected action 0:
  4,096 / 4,096
```

### S9

Both designs selected the population-optimal action in all locked replicates:

```text
passive correct-best / top-2:
  1 / 1

Neyman correct-best / top-2:
  1 / 1

passive / Neyman mean selection regret:
  0 / 0

passive D-hat mean:
  0.050232142015460016

Neyman D-hat mean:
  0.05010110016606279

passive sample variance:
  0.00012466963074279542

Neyman sample variance:
  0.000048891890271740703
```

The frozen result explicitly sets:

```text
universal_active_superiority_claim:
  false
```

The fixed S9 law therefore validates the estimator and variance-design
contract; it does not establish a universal active-testing advantage.

## Proof-Candidate Freeze

Exactly seven self-contained candidates are frozen:

| Theorem | Candidate disposition | Formal status | Candidate SHA-256 |
|---|---|---|---|
| T1 | `PROPOSED_PROOF` | `OPEN` | `57ef99b65846ff62dc1471de311433c8a605774721bd0dbaf89c8a1bd0de98cf` |
| T2 | `PROPOSED_COUNTEREXAMPLE` | `OPEN` | `417a7496bebff41a8ca1c0240c5a5c7d690124b8fe33c49194d4c51458f2b6fc` |
| T3 | `PROPOSED_PROOF` | `OPEN` | `cdeda8e9feae09a13617dabc82735633fe39affe7fcd7b9a3aca628e957088d1` |
| T4 | `PROPOSED_PROOF` | `OPEN` | `9238f914c1bb5efbf5980050b02e0e72e20c934ea30f1287967f2ecc70755d03` |
| T5 | `INCOMPLETE_OPEN` | `OPEN` | `a6a38358dc3bf1bc2b39e1fda8bdece0a2a416b7627bb16bd671047724d90188` |
| T6 | `PROPOSED_COUNTEREXAMPLE` | `OPEN` | `d9325fab96d634b5164a3528c0bb9da484d4b20714116027f5bb39d5826b917f` |
| T7 | `PROPOSED_PROOF` | `OPEN` | `1316236128ae0389f681459ff28dfa99a248b3ad73357208821bb3220476a857` |

The check class is:

```text
PROOF_CANDIDATE_SCHEMA_AND_INTERNAL_CONSISTENCY
```

It is not independent proof review. C85T performs no formal status transition.

## Semantic Replay

The frozen semantic receipt reports:

```text
status:
  SEMANTIC_REPLAY_PASS

scenario results:
  11

S6/S7 logical rows:
  8,192

S9 logical design rows:
  8,192

S9 digest rows / RNG replays:
  4,096 / 4,096

proof candidates:
  7

formal OPEN statuses:
  7

protected counters zero:
  true
```

The post-execution collector additionally replayed all manifest path/size/hash
rows, the lifecycle chain, the completion chain, the external/copy receipt,
and final-root coverage. It did not invoke registered exact or Monte Carlo
dispatchers.

## Regression Verification

```text
focused:
  409 passed, 1 deselected

C65:
  1,020 passed, 1 skipped, 4 deselected

C23:
  1,431 passed, 1 skipped, 4 deselected

full OACI:
  2,355 passed, 1 skipped, 4 deselected

accepted stderr:
  0 bytes for all suites
```

The additional post-execution deselection is the C85TR2 readiness-only
assertion that the authorization record does not exist. The authorization is
now expected and consumed. No lock-bound test was edited.

Two non-accepted operational invocations are disclosed in the regression
report: a wrong-Python collector import and an `srun` rejection caused by a
stale inherited Slurm ID. Neither ran tests or registered scenarios.

## Final Red Team

```text
checks: 80
PASS:   80
FAIL:    0
```

The final red team replays authorization, scheduler identity, lifecycle,
atomic publication, artifact bytes, semantic counts, proof governance,
protected counters, and regression evidence.

## Protected Scientific Boundary

```text
real project data access:    0
EEG / labels / logits:       0 / 0 / 0
training / forward / GPU:    0 / 0 / 0
active acquisition:          0
theorem-status transitions:  0
C85V authorized:             false
C85E authorized:             false
new data/model zoo:          false
manuscript modified:         false
```

C85T is a synthetic/statistical-decision result only. It makes no new empirical
claim about C84 datasets and does not modify C84-D or C84-L4.

## Operational Disclosure

At final reporting, `squeue` showed two unrelated user jobs:

```text
897842  bash          cpu-high  RUNNING
899527  targetx-full  cpu-high  RUNNING
```

Neither was submitted or changed by C85T. No active C85T V3 job remained.

## Next Boundary

The next possible milestone is a separately approved read-only proof review:

```text
C85V
```

C85V may audit proof candidates and issue theorem-status verdicts under a new
protocol. It must not rerun the Monte Carlo benchmark. C85T completion does not
authorize C85V, C85E, active acquisition, real data, new data/model zoos, or
manuscript changes.

## Completion Gate

```text
C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED
```

