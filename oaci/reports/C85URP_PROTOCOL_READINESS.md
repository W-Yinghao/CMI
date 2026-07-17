# C85URP Candidate-Utility Reconstruction Readiness

## Final Gate

```text
C85U_HELD_EVALUATION_CANDIDATE_UTILITY_RECONSTRUCTION_LOCKED_READY_FOR_PI_AUTHORIZATION
```

C85URP implemented and locked a future protected reconstruction path. It did
not execute C85U, open real evaluation-label rows, open target NPZ payloads,
read Q0 shards, or compute a real candidate utility.

The existing C85EP blocker remains historically correct. C85U is a new
prospective input-production milestone; it does not make C85EP retroactively
ready.

## Authoritative Chronology

```text
C85EP blocker HEAD:
  c470084d94910bfe7290a8565db30d058d5b3de6

C85U protocol commit:
  ebe158c9e929f67423a9ebdc3cea7c6ea5c16c9a

C85U protocol SHA-256:
  c9ed7081cf8cb1a6c8a05181d1660da2015b4e1716a05c8916f7fe5b09efc160

C85U implementation commit:
  df100e2e77c5749030e2931bb7752973258823bd

C85U execution-lock commit:
  3b4fa48ee2d4f75ff8ba2191dc7d8593237dc82f

C85U execution-lock SHA-256:
  923c6bee2171f0bedcc3f883058759d368bdb49eb272cbbfa80974e98b632fe1
```

The protocol was committed and pushed before metadata discovery and before
implementation. Metadata discovery then used only manifests, committed
registries, path existence, and file sizes. The implementation commit preceded
the execution-lock commit.

## Frozen Inputs

```text
C84F complete-field manifest:
  cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8

C84S V5 analysis lock:
  030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846

C84S selection freeze:
  30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4

C84S scientific result:
  5590f85c3552ec0176a015e34296059a950dd2c5853a51aa140657cf53d79ee7

C84S result manifest:
  516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5
```

Stage-A evaluation identity:

```text
evaluation seal:
  54e06dff60d80255631dc4faa20c8c7db651f2af8fc5415671dd9ab6681b5502

evaluation-view manifest:
  6fad247629eb48340a4badf9ab1a0669652757a58216e46826e4dfd8bfd608bd

evaluation label table, bound but not opened:
  ea76c34663edac1e6e7e844fee6af3f06058aaaf3846febda1dff94df343a371
  394,109 bytes
  4,848 rows
```

Field registry:

```text
field descriptors:          1,944
target artifacts:           1,944
target-artifact bytes:      48,018,748,054
candidate zoos:             24
target contexts:            944
candidates per context:     81
future utility rows:        76,464
```

The 1,944 target paths, sizes, expected hashes, sidecar identities, canonical
candidate order, and 944 context identities are frozen in
`c85urp_tables/`. Their payloads were not opened.

Future U2 inputs were bound from their parent manifests without opening them:

```text
candidate_ranks.csv
fixed_default_selections.csv
q0_selection_shard_index.csv
method_context_decisions.csv
```

## Historical Formula

The lock binds the existing byte identities for:

```text
oaci.multidataset.c84s_evaluation.context_candidate_utility
oaci.multidataset.c84s_q0_budget.endpoint_metrics
oaci.multidataset.c84s_q0_budget.midrank_percentile
```

For each 81-candidate context, U1 will compute held-evaluation balanced
accuracy, NLL, and ECE; orient their within-context midrank percentiles; and
take the arithmetic mean. The canonical best action is the first canonical
argmax. Standardized regret uses the historical `1e-15` zero-spread rule.

No formula rewrite, metric cleanup, alternative tie rule, or endpoint
reweighting is permitted.

## Stage U1

The future U1 subprocess receives only:

```text
receipt-validated authorization context;
immutable evaluation-label seal/view;
immutable C84F target artifacts;
frozen field/context descriptors.
```

It receives no construction labels, Stage-B scores/ranks, fixed selections,
Q0 shards, method decisions, inference tables, or taxonomy.

It writes one non-object compressed NPZ per context and one complete row index:

```text
context artifacts:          944
rows per artifact:           81
candidate index rows:    76,464
```

Output fields include identities, bAcc/NLL/ECE, oriented midranks, composite
utility, canonical order/rank, standardized regret, top-1/5/10 indicators, and
input/metric/utility digests. Outputs explicitly exclude labels, logits,
probabilities, EEG, and other source arrays.

Every NPZ is reloaded before acceptance. Float arrays are little-endian
`float64`; metric and utility replay tolerance is `1e-12`; identity, dtype,
shape, SHA-256, midrank, order, and top-k replay is exact. The complete U1 root
is published only after replay derives 944 contexts and 76,464 rows from the
actual staged artifacts and index.

## Stage U2

U2 runs in a separate subprocess after the atomic U1 manifest exists. It
receives only:

```text
U1 utility field;
frozen Stage-B ranks/fixed actions/Q0 action shards;
historical method_context_decisions.csv.
```

It receives no label, label-root, target artifact, or logit argument. It does
not import analysis, inference, max-T, LOTO, frontier, or taxonomy code.

U2 reconstructs and compares exactly:

```text
method-context rows:         18,432
selected utility
standardized regret
top1 / top5 / top10
selected regime
```

Finite Q0 budgets replay the existing 2,048 chain action orders. No seed,
sample, score, chain, or selection is regenerated. B0, B5, fixed defaults,
score-method orders, and Q0 FULL use their frozen historical semantics. A U2
mismatch leaves the U1 artifact frozen but not accepted for C85E.

## Authorization And Runtime

The execution lock is unauthorized. A future exact statement is required:

```text
授权 C85U
```

The future record must use schema
`c85u_direct_pi_authorization_record_v1`, bind one exact content-addressed
output root, and set C85E/C86/active/real-data-zoo/manuscript fields false.
Authorization is consumed once through an external `O_CREAT|O_EXCL` receipt.

Before consumption, the runtime replays the lock, all 45 bound repository
objects, clean `oaci` HEAD/origin identity, exact Python/NumPy/SciPy identities,
metadata registries, output policy, and storage envelope. Evaluation-label and
target-artifact bytes are first hashed only after authorization consumption.

Locked resources:

```text
partition:       cpu-high
CPU:             48
RAM:             128 GiB
GPU:             0
wall:            2 hours
output maximum:  2 GiB
```

## Protected Boundary At Readiness

```text
evaluation-label rows opened:       0
target-artifact payloads opened:     0
target-sidecar payloads opened:      0
Q0 shards opened:                    0
direct C84S tables opened:           0
real candidate utilities computed:   0
real historical decisions replayed:  0
authorization records created:       0
C85U executions:                     0
C85E/C86 executions:                 0 / 0
training / forward / GPU:             0 / 0 / 0
```

## Immutable Results

```text
C84 primary:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

C84 frontier:
  C84-L4

C85 statuses:
  T1/T3/T4/T7 PROVED
  T2/T6 COUNTEREXAMPLE
  T5 OPEN
```

C85URP changes none of these results.

## Validation

```text
focused: 394 passed
C65:   1,067 passed, 1 skipped, 5 deselected
C23:   1,478 passed, 1 skipped, 5 deselected
full:  2,402 passed, 1 skipped, 5 deselected
```

All four accepted stderr files are empty. The new C85URP files independently
passed `19/19` after the lock commit.

## Continuation Boundary

C85U is ready for PM review and fresh authorization, but not authorized. A
successful future execution must stop at:

```text
C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED
```

It will not authorize C85E. A later C85EP2 must replay U1 and U2 and create a
separate C85E lock. C86, active acquisition, new data/model zoos, and manuscript
work remain unauthorized.
