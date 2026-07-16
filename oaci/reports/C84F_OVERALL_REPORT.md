# C84F Overall Report

## 1. Decision

C84F has completed the multi-dataset, dual-level, fixed-zoo model field and its
complete label-free target instrumentation. The authoritative final gate is:

```text
C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED
```

The atomic complete-field manifest SHA-256 is:

```text
cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8
```

This is an engineering field-completion gate, not a scientific result. C84F
did not compute target accuracy, selector scores, Q1/Q2, label-budget
statistics, level effects, cross-dataset recurrence, or external validity.
C84S is not authorized and has no execution lock.

## 2. Epistemic And Scope Boundary

C84 is a separate prospective external-validity branch built after the C83
single-dataset evidence freeze. The latest valid scientific result remains
C82-D. C84F creates the frozen candidate and label-free instrumentation field
needed for a later C84S protocol; it does not itself open a scientific endpoint.

The completed field covers:

```text
datasets:               Lee2019_MI, Cho2017, PhysionetMI
task:                   left_hand versus right_hand motor imagery
source panels:          A and B
training seeds:         5 and 6
levels:                 0 and 1
candidate zoos:         24
training phases:        72
candidate units:        1,944
target subjects:        118
target contexts:        944
candidate-contexts:     76,464
```

Level 0 uses the complete locked 12-subject source-training panel. Level 1 is
the target-independent fixed-panel source-support stress that deletes one
registered source subject x `left_hand` cell before plan construction. The
registered deleted subjects are Lee A/B `31/16`, Cho A/B `17/37`, and
Physionet A/B `103/109`. No target-specific retraining or outcome-selected cell
was used.

## 3. Interface Identity

The field uses interface `C84_LEFT_RIGHT_20CH_160HZ_0_3S_V2`:

```text
channels:
  FC5 FC3 FC1 FC2 FC4 FC6
  C5 C3 C1 Cz C2 C4 C6
  CP5 CP3 CP1 CPz CP2 CP4 CP6

montage SHA-256:
  988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04

sample rate:            160 Hz
epoch:                  half-open [0.0,3.0), 480 samples
interpolation:          forbidden
Fz substitution:       forbidden
zero filling/masks:     forbidden
```

Physionet subject 88 remains prospectively excluded. Lee unlabeled online runs
remain excluded. The structural target-y slot was not indexed, converted,
represented, hashed, summarized, or logged by the target-stage implementation.

## 4. Protocol, Lock, And Authorization Identity

The successful target-stage execution was bound to the following additive
objects:

```text
C84FR2 repair protocol commit:
  27fc479ecd4131ceb4f79982cb0890f517709d2e
C84FR2 repair protocol SHA-256:
  4480530f05f0814b712edf23a51f08e64820ddea220d6b26f4dc7bb0f2a541bf

target instrumentation V2 commit:
  b527b82950690d09e73e5f3468d994cf11b56413
target instrumentation V2 SHA-256:
  2387b3a1a81dd061b7cc723820d41829e0237e5be0f243bc423422a173f72671

C84FR2 implementation commit:
  a64c6815ad512d0b8bc7158cd61ff2147f24948a
C84FR2 execution-lock commit:
  34eb5efad16f1f8a320b08435363939089c8037a
C84FR2 execution-lock SHA-256:
  f0c369ee273352b47e36ce426108d78a2d4193180afbf0aecfbbeb3d4980ba47

authorization commit:
  fc40914d104bde8522c82106b615f2e29ad38e9c
authorization record SHA-256:
  b05ba5dd5c6a6393038d491a5b1407c7a973b0c20494963e44b91311b482a05b
authorization consumption SHA-256:
  d25e8047a8245c85640b4c06e80c5d06a65fb3db39463a0a431dd08b74fab0e3

initial result/report commit:
  d3eb1919eb29c028d4570f605fd6bc1c05bfbc5e
```

The V2 lock also replays the C84FL2 reconciliation, field V7, full-field V2,
external V3, scientific V3, level-1 intervention, level-1 numerical repair, and
C84FR1 canonical-registry repair protocols. It binds 38 repository objects, 19
implementation files, four loader-source identities, 7,776 model artifacts,
2,430 canary files, and all 1,944 operative unit identities.

The direct authorization covered only target-stage numerical replay repair. It
explicitly set model retraining, target labels, scientific metrics,
same-label-oracle access, and C84S to false.

## 5. Accepted Engineering Reuse

Two prior canaries supplied reusable model/state/source-audit objects:

| Canary | Job | Level | Units | Phases | Complete manifest SHA-256 |
|---|---:|---:|---:|---:|---|
| C84C | 895441 | 0 | 243 | 9 | `530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b` |
| C84L1C | 896066 | 1 | 243 | 9 | `3cf1366ccf40efc82a6bb2ffef56045e83c0f0e9670429973f23252371ad1c18` |

The combined reuse was 486 units, 18 training phases, and 2,430 replayed
external files. Reusable objects were candidate identities, checkpoints,
optimizer states, training/genealogy sidecars, and strict-source audit
artifacts. The six canary target contexts were replay witnesses only; they were
not spliced into the final target field.

## 6. Complete Model Field

Job `896185` completed the full model field before its later target-registry
failure. The accepted model manifest is:

```text
model-field manifest SHA-256:
  d8931b81a3d68f4b1e098ac6e3ede3cd44cdb6c70cdef9f18a76e0a8c62ecdb2

candidate units:                    1,944 / 1,944
training phases:                       72 / 72
checkpoint/optimizer/sidecar/audit: 7,776 files
level 0 / level 1:                   972 / 972
C84C / C84L1C reused:               243 / 243
new C84F units:                   1,458 / 1,458
unique unit IDs:                  1,944 / 1,944
```

Each dataset contributes 648 units: 324 per level, 324 per panel, and 324 per
seed. The regime counts are 24 ERM, 960 OACI, and 960 SRC units. Model-field
freeze counters record zero training target rows, zero training target labels,
zero source-audit rows used in training, and zero target-outcome retention or
retry. Job `896185` accessed no additional target subject before the atomic
model-field freeze. The six earlier C84C/C84L1C canary contexts are separately
disclosed historical engineering exceptions and never affected retention.

## 7. Preserved Failure And Repair Lifecycle

### 7.1 Job 896185: target-registry ordering defect

The originally authorized C84F job completed all 72 phases and froze all 1,944
model units. After that barrier it loaded 118 target arrays, then stopped before
freezing any target-registry row or candidate forward because Python attempted
to sort dictionaries directly:

```text
TypeError: '<' not supported between instances of 'dict' and 'dict'
operation: sorted(dict(row) for row in view.raw_files)
```

The accepted disposition was
`C84F_TARGET_REGISTRY_CANONICAL_ORDER_REPAIR_REQUIRED_NO_OUTCOME_CONTAMINATION`.
The model field and raw-input manifest were retained; the consumed
authorization and failed root were not reusable. Target-y, target labels,
selector scores, scientific metrics, and oracle access were all zero.

### 7.2 Job 896550: cross-backend persisted-matmul defect

C84FR1 canonicalized the registry and froze the exact 118-subject, 9,621-row
label-free target identity. Its target instrumentation then stopped on unit
`c84l1_00cb2c89efa87efe281dbb9229c63e53` at a legacy NumPy float32
reconstruction of `2.193450927734375e-05` versus a `2e-5` gate.

The failed root contained six NPZ files and five context indices, with five
completed artifact counters and 268 partial candidate-context slices. All 11
objects were retained as evidence and rejected from final-field reuse. No model
was retrained and no target label or scientific outcome was accessed.

### 7.3 C84FR2 repair

C84FR2 did not widen `2e-5`. It separated:

1. same-GPU/PyTorch functional classifier identity at `1e-6`;
2. exact dtype/shape/byte-digest persistence for all registered arrays;
3. saved-output replay at `1e-6`;
4. finite CPU/NumPy reconstruction diagnostics with no gate authority.

The replacement entrypoint is target-only and has no training import,
optimizer creation, checkpoint write, retention logic, selector import, or
scientific-analysis callable.

## 8. Pre-Submission Replay And Resources

The final no-data preflight passed before submission:

```text
bound repository objects:          38 / 38
protocol sidecars:                 10 / 10
model units:                    1,944 / 1,944
model artifacts:                7,776 / 7,776
canary units:                     486 / 486
canary files:                   2,430 / 2,430
target subjects / rows:           118 / 9,621
historical partial objects:         11 rejected
loader source files:                 4 / 4
HEAD == origin/oaci:                    true
active C84 jobs through squeue:            0
```

The output filesystem had `2,866,721,849,344` available bytes and
`5,599,065,414` available inodes. The fresh content-addressed run root did not
exist. The requested 5-hour allocation was below the 5-hour sharding threshold
and had substantial headroom over the measured target-stage estimate.

Two operator-only preflight summary wrappers failed after the protected guard
had passed because they assumed incorrect return shapes. Neither consumed
authorization, imported the protected data path, or accessed EEG. The empty
root created by the first availability check was removed with `rmdir`; the
final no-data preflight passed. These were reporting-wrapper failures, not
target-stage application attempts.

## 9. Scheduler And Runtime

The exact locked target-only entrypoint ran as job `897048`:

```text
partition / node:        V100 / node11
GPU:                     1 V100
CPU / memory:            8 / 64 GiB
wall request:            05:00:00
start:                   2026-07-15T12:56:16.073594400Z
complete:                2026-07-15T14:58:19.076017890Z
application wall:        7,322.999471096 s (about 02:02:03)
```

The locked command was:

```text
/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python \
  -m oaci.multidataset.c84fr2_target_stage run-real \
  --authorization-record \
  /home/infres/yinwang/CMI_AAAI_oaci/oaci/reports/C84F_TARGET_STAGE_NUMERICAL_REPLAY_PI_AUTHORIZATION_RECORD.json \
  --output-root \
  /projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2
```

The runtime identity was Python 3.13.7, torch 2.6.0+cu124, MOABB 1.5.0,
MNE 1.11.0, and chardet 5.2.0 with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, and all registered BLAS
thread counts set to one.

A100, H100, and L40S were not substituted because the execution lock bound one
V100 and the prospective duration was below five hours. Changing GPU
architecture would have required a new lock and cross-architecture canary
review. The actual runtime confirmed that sharding was unnecessary.

Stage timings were:

| Stage | Seconds |
|---|---:|
| guard, imports, and frozen-input barrier | 12.452210921 |
| target X and exact registry replay | 1,240.043069406 |
| complete V2 target instrumentation | 5,806.071385733 |
| aggregation and atomic manifest publication | 264.432805036 |

Scheduler monitoring used `squeue` plus the application ledger and partial
manifest. `sacct` was not used, and no scheduler terminal state was inferred.
The application ledger's atomic `complete` event is the execution authority.

## 10. Label-Free Target Input Freeze

The accepted target-input identities are:

```text
raw-input manifest SHA-256:
  9539747e903dfe67295ee04a97441b85c0bb2179c9ef1bd2177788865e0ba5fd
trial-registry SHA-256:
  52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8
```

| Dataset | Subjects | Trial rows | Raw files | Raw bytes |
|---|---:|---:|---:|---:|
| Lee2019_MI | 22 | 2,200 | 44 | 26,461,699,033 |
| Cho2017 | 20 | 4,000 | 20 | 4,082,102,946 |
| PhysionetMI | 76 | 3,421 | 608 | 1,357,354,752 |
| **Total** | **118** | **9,621** | **672** | **31,901,156,731** |

Every row records only dataset, target subject, stable trial ID, session, run,
interface/montage identity, sample rate/count, finite-value status, and raw
path/size/hash identity. It contains no class label, label hash, label count, or
label-derived availability field.

The earlier cache audit found zero repeated download URLs within job `896185`.
Lee and Cho reused readable shared caches. The shared Physionet cache was not
readable by the executor, so the separately owned cache held the 864 files
required by the complete source/target workflow. C84FR2 replayed the frozen raw
identities rather than redefining the dataset inputs.

## 11. Complete Target Instrumentation

| Dataset | Candidate units | Targets | Trial rows/unit | Contexts | Candidate-context slices |
|---|---:|---:|---:|---:|---:|
| Lee2019_MI | 648 | 22 | 2,200 | 176 | 14,256 |
| Cho2017 | 648 | 20 | 4,000 | 160 | 12,960 |
| PhysionetMI | 648 | 76 | 3,421 | 608 | 49,248 |
| **Total** | **1,944** | **118** | **9,621 registry rows** | **944** | **76,464** |

The complete V2 field contains:

```text
target NPZ artifacts:             1,944 / 1,944
context/digest sidecars:          1,944 / 1,944
field descriptors:                1,944 / 1,944
candidate-context slices:        76,464 / 76,464
canary-unit witnesses:              486 / 486
```

Each NPZ has exactly 21 registered fields:

```text
unit_id, dataset, panel, training_seed, level, level_intervention_id,
regime, epoch, trajectory_order, target_subject_id, target_trial_id,
session, run, logits, probabilities, z, Wz_plus_b,
classifier_weight, classifier_bias, repeat_logits, repeat_z
```

Every field passed exact pre-write/post-reload dtype, shape, byte SHA-256,
scalar identity, and trial-order replay. Unknown, missing, object-dtype, or
nonfinite fields were forbidden.

## 12. Numerical Identity

Strict functional and saved-output gates all passed:

| Check | Observed maximum | Gate |
|---|---:|---:|
| same-GPU/PyTorch direct classifier vs model logits | 0 | `1e-6` |
| saved `Wz_plus_b` vs saved logits | 0 | `1e-6` |
| saved softmax replay | `1.1920928955078125e-07` | `1e-6` |
| repeat logits | 0 | `1e-6` |
| repeat z | 0 | `1e-6` |

Cross-backend values were finite diagnostics only:

| Backend | Maximum | Maximum p95 | Maximum p99 |
|---|---:|---:|---:|
| CPU PyTorch float32 | `4.57763671875e-05` | `1.0728836059570312e-05` | `1.621246337890625e-05` |
| NumPy float32 | `4.1961669921875e-05` | `1.0967254638671875e-05` | `1.6689300537109375e-05` |
| NumPy float64 | `1.1513513614502813e-05` | `3.5327511082261455e-06` | `5.1939440638593156e-06` |

These diagnostics did not affect retention, retry, model state, candidate
identity, thresholds, or interpretation. No runtime tolerance was widened.

## 13. Storage And Artifact Identity

The successful external root is:

```text
/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2/
  lock_f0c369ee273352b47e36
```

| Object class | Files | Bytes |
|---|---:|---:|
| target NPZ artifacts | 1,944 | 48,018,748,054 |
| context/digest sidecars | 1,944 | 24,209,257 |
| top-level manifests and ledgers | 8 | 15,727,706 |
| **Complete root** | **3,896** | **48,058,685,017** |

Important top-level identities:

| Object | Bytes | SHA-256 |
|---|---:|---|
| authorization consumption | 1,037 | `d25e8047a8245c85640b4c06e80c5d06a65fb3db39463a0a431dd08b74fab0e3` |
| execution attempt ledger | 6,109 | `1cb6bbcbeee6f9c958c8d67ed256d3f47c7a6313f9613b6a1d54fe052777ce2e` |
| partial/final artifact ledger | 684,762 | `323190163492751ffc53123a96bee0fe513452f6a8753709c4e65719982e56e6` |
| raw-input manifest | 156,940 | `9539747e903dfe67295ee04a97441b85c0bb2179c9ef1bd2177788865e0ba5fd` |
| target trial registry | 8,544,261 | `52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8` |
| complete-field manifest | 6,334,399 | `cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8` |

No raw EEG, model checkpoint, optimizer state, NPZ cache, or external payload
was added to Git.

## 14. Protected State

The completed attempt and complete manifest record:

```text
model retraining:                   0
training phases started/completed: 0 / 0
source EEG/source labels:           0 / 0
target-y operations:                0
target-label fields:                0
target construction labels:         0
target evaluation labels:           0
same-label oracle:                   0
selector scores:                     0
scientific statistics:              0
target scientific metrics:          0
C84S authorized:                 false
```

Target instrumentation had no training callable and did not modify any model,
optimizer, candidate retention decision, or model-field manifest.

## 15. Logs And Warnings

The successful stdout is 59,895 bytes with SHA-256
`d62ba398d3b15d5d7181bb7fa80c5de346d91fb79cde77a602e3cd33b1545b74`.
The stderr is 1,900 bytes with SHA-256
`460688ec39ffbe001d7e3fb2f710dabce78de43448ef3e9ed41aab2f78766700`.

Stderr contains exactly 20 identical MOABB/MNE notices:

```text
Trials demeaned and stacked with zero buffer to create continuous data -- edge effects present
```

The notices are disclosed nonblocking loader warnings. Stderr contains zero
traceback, exception, runtime-failure, numerical-failure, or authorization
failure marker.

## 16. Independent Post-Execution Replay

A separate streaming audit read the complete target field after atomic
publication. It verified:

```text
NPZ path/size/SHA identities:      1,944 / 1,944
sidecar path/size/SHA identities:  1,944 / 1,944
all 21 persisted field digests:     PASS
exact directory membership:         PASS
target contexts:                    944
candidate-context slices:        76,464
canary witnesses:                   486
protected counters:                 zero/false
```

The artifact replay ledger SHA-256 is
`16c89c8b845383e27dc9787a02812b458495f237217d08d61bd38c22f9bee53f`.
The final bound-input replay ledger SHA-256 is
`1431fb2fb9dad7f6e1becc0cd8e92c6eaf350cf683ea681fa6672c2259fb61c2`.
Both report `PASS`.

## 17. Validation, Red Team, And Regression

C84FR2 readiness had 9/9 synthetic fixtures and 56/56 readiness red-team
checks. The post-execution final report red-team passed 68/68.

Post-execution CPU regression results were:

| Suite | Result | Pytest time | Stderr bytes |
|---|---|---:|---:|
| focused | 30 passed | 1.48 s | 0 |
| C65 | 758 passed, 1 skipped, 3 deselected | 608.98 s | 0 |
| C23 | 1,169 passed, 1 skipped, 3 deselected | 512.97 s | 0 |
| full OACI | 2,093 passed, 1 skipped, 3 deselected | 1,634.05 s | 0 |

The conditional skip is the finalized C78F field test. The three explicit
deselections are historical C79 authorization-state fixtures whose premise is
incompatible with later committed authorization records. No C84FR2
implementation, persistence, isolation, or complete-field test was deselected.

## 18. Repository And Operational Closeout

At the initial result closeout:

```text
HEAD == origin/oaci:
  d3eb1919eb29c028d4570f605fd6bc1c05bfbc5e

oaci worktree:
  clean

active C84 jobs through squeue:
  0

C84S lock or authorization record:
  0

tracked Git files over 50 MiB:
  0
```

The outer `/home/infres/yinwang/CMI_AAAI` worktree contains pre-existing
untracked project material and the preserved C84FR2 execution logs. It was not
cleaned or rewritten. The operative `CMI_AAAI_oaci` worktree is the repository
state relevant to this result.

## 19. Evidence Boundary And Next Stage

C84F establishes only that a complete, replayed, label-free multi-dataset field
exists under the locked engineering contract. It does not establish:

```text
target accuracy
zero-label selector performance
Q1 or Q2
label-budget frontier
level effect
same-method cross-dataset recurrence
external validity
deployment utility
```

The next permissible milestone is a separately designed C84S scientific
adapter and analysis execution lock. Only after PM review and fresh direct PI
authorization may C84S provision physically disjoint construction/evaluation
labels and compute registered scientific endpoints. No C84F authorization or
field-completion gate carries forward automatically.

## 20. Authoritative Report Set

The complete reporting set is:

```text
oaci/reports/C84F_OVERALL_REPORT.md
oaci/reports/C84F_OVERALL_REPORT.json
oaci/reports/C84F_OVERALL_REPORT.sha256
oaci/reports/C84F_MULTI_DATASET_DUAL_LEVEL_FIELD.md
oaci/reports/C84F_MULTI_DATASET_DUAL_LEVEL_FIELD.json
oaci/reports/C84FR2_EXECUTION_RESULT_IDENTITY.json
oaci/reports/C84F_FINAL_REPORT_RED_TEAM.md
oaci/reports/C84F_REGRESSION_VERIFICATION.md
oaci/reports/OACI_EEG_DG_PROJECT_MEMORY_THROUGH_C84F.md
oaci/OACI_CODEX_HANDOFF.md
```
