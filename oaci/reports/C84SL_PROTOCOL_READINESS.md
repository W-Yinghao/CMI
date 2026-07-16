# C84SL Protocol Readiness

## Final decision

```text
C84S_MULTIDATASET_LABEL_VIEWS_SELECTION_INFERENCE_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION
```

C84SL completed the no-real-label implementation and lock-readiness milestone.
It did not provision a real target label, compute a real selector score, inspect a
scientific outcome, run training or forward inference, use a GPU, or authorize
C84S.

The operative C84S lock is:

```text
path:
  oaci/reports/C84S_ANALYSIS_EXECUTION_LOCK_V2.json

lock commit:
  33075c97afd87f05d2856463c43be3246d83f95c

SHA-256:
  94c896f0f00c53441095da6225f9ac574eb4a9baa904821a5dab3f11ea76f75c

status:
  LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED
```

## Scope and evidence boundary

C84SL implemented the future C84S label-view, selection-freeze, held-evaluation,
inference, robustness, taxonomy, and atomic result-freeze paths. All executable
calibration used synthetic fixtures and frozen schemas.

Protected counters at lock creation were:

```text
real target-label access:       0
real selector scores:           0
real scientific statistics:     0
training:                        0
forward inference:              0
GPU execution:                  0
same-label oracle access:       0
C84S authorization records:     0
```

C84SL does not establish external validity or any C84-A--E/C84-L1--L4 result.
Those outcomes remain unavailable until a separately authorized C84S execution.

## Frozen-field identity replay

The implementation and V2 lock bind the accepted C84F field without opening its
label views:

```text
C84F final HEAD entering C84SL:
  bcb7ee9237e9451f23fab5ea9f64fa48d89be5a0

C84F complete-field manifest:
  cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8

C84F overall report Markdown:
  f80089fa03a64da5b2137e005d86eec2b282b4ab5ea33206f2f2a96ac321fe0c

C84F overall report JSON:
  edb6ffb73e2f65ce56102f75abbe6ee447ca9dbf1cdddb7631f0ecbfa0b30f47

model-field manifest:
  d8931b81a3d68f4b1e098ac6e3ede3cd44cdb6c70cdef9f18a76e0a8c62ecdb2

target raw-input manifest:
  9539747e903dfe67295ee04a97441b85c0bb2179c9ef1bd2177788865e0ba5fd

target trial registry:
  52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8
```

Bound field counts:

```text
datasets:                         3
candidate units:              1,944
target artifacts:             1,944
context/digest sidecars:       1,944
external artifact hashes:      3,888 / 3,888 replayed
target subjects:                 118
target trial-registry rows:     9,621
target contexts:                  944
candidate-context slices:      76,464
```

The committed target-artifact registry contains only artifact identity,
path/size/hash, and schema evidence. C84SL did not inspect target predictions or
derive a candidate score from them.

## Prospective protocol chronology

The scientific-analysis protocol V3 was preserved unchanged:

```text
C84 scientific-analysis protocol V3 SHA-256:
  bf6c7f718413b4b2ac2ad9786aa2e47dc045a536e7237d5d8c0464b6598130b8
```

Executable semantics were registered before their implementation:

| Object | Commit | SHA-256 |
|---|---|---|
| C84S operationalization protocol | `5d7f07cde218bff48f504bffb7d9bd8fd8bfa5ff` | `abf56676901bd7e5f484ffe4f4bb49de625d5c7b87cc34c0e3ab2bdb39361c5e` |
| End-to-end result-freeze repair | `e2669cd33e357ca9ebd19275fae2344faae70094` | `b2d52a3bfdcb89b8a8db5d1a5501fb7b24a22ad860dbe1d6da2c5e6d77ca189c` |
| Label-frontier stability clarification | `0a1ecf31ff8a534310c0560169f75529ac1462bf` | `020251ded9dbe4688ef08e9854875fc06b789120f4697c661b94f96eeef66fca` |
| Repaired implementation and table registry | `56583ea386551e8310d18cd805e14fdf92eceaa6` | bound file-by-file in the V2 lock |
| Operative V2 execution lock | `33075c97afd87f05d2856463c43be3246d83f95c` | `94c896f0f00c53441095da6225f9ac574eb4a9baa904821a5dab3f11ea76f75c` |

The initial lock at commit `c2f75c265b7abaa09c175524a9ae0e2a650173f7`
is preserved with SHA-256
`e17e4da14b60ac77ca0ec8bec80a2ca249cda014baf5460cfd64627294f2047b`.
It was never authorized and is explicitly non-operative because a pre-label
audit found that it did not bind a complete production Stage-C result-freeze
path. The repair was additive; no historical lock or protocol was rewritten.

## Physical label-view and process isolation

The future runtime is split into three one-way stages:

1. Label provisioning aligns labels to the exact 9,621-row frozen trial
   registry and writes physically separate construction and evaluation roots.
   Candidate artifacts are unavailable to this process.
2. Selection receives source-audit and target-unlabeled artifacts, plus only the
   construction view for Q0. It has no evaluation path or descriptor and
   atomically freezes all scores, ranks, selected IDs, top-k IDs, input-access
   ledgers, and Q0 chain identities.
3. Held evaluation receives the immutable selection manifest and the evaluation
   view. Its public interface cannot modify candidate scores, ranks, or selected
   IDs.

The split is fixed by
`SHA256(C84_TARGET_SPLIT_V1|dataset|subject|trial_id)`, class-stratified per
dataset and target. Construction/evaluation overlap must be zero and each side
must retain at least eight labels per class. No same-label oracle path exists.

Static isolation auditing passed 16/16 checks. In particular, the label
provisioner has no candidate-artifact import, selection has no evaluation-view
path, evaluation cannot call selection, and the C84S implementation has no
training, checkpoint-forward, optimizer, or GPU callable.

## Registered methods and Q0

The lock replays method-registry SHA-256
`ef48ecf7fcc55188b78b0878d86f07f6239fe4f6c88bbc854829b3a1c7a1a120`.
The operative comparison includes fixed controls, S1, and the six zero-label
primaries U5/U7/U11/U13/U14/U15. No method, score direction, prior,
temperature, layer, tie rule, or family representative can be changed at
execution.

Q0 is fixed to:

```text
primary budgets:                 [1,2,4,8,FULL]
Lee/Cho secondary budgets:       [16,32]
RNG:                             PCG64
chains:                          2,048
stream:                          SHA256(C84_Q0_V1|dataset|target|chain)
scientific sample:               target subject
```

Finite budgets use paired nested class-stratified subsets across panel, seed,
level, and candidate. `FULL` uses the canonical construction-view trial order
and is deterministic across chains, including its sample, score, metric, and
top-k digests. Monte Carlo chains cannot enter target-cluster inference.

## Evaluation, inference, and taxonomy

The production Stage-C implementation validates exactly 18,608 mixed
method-context rows before opening a final result path. It computes the locked
balanced-accuracy/NLL/ECE utility, standardized regret, selected utility,
source-relative gain, top-1/top-5/top-10, coverage, selected regime,
catastrophic failures, and semantically applicable measurement diagnostics.

Aggregation is equal-weighted:

```text
within target and level:       four panel x seed cells
within target primary:         eight panel x seed x level contexts
within dataset:                target-subject effects
principal scientific cluster:  target subject
```

Q1/Q2 use 65,536-draw target-cluster Rademacher/max-T inference over the six
fixed zero-label methods inside each dataset. No pooled three-dataset p-value is
implemented. Level decisions precede averaging, panel/seed gates remain
explicit, and LOTO preserves method identity at 17/22 Lee, 15/20 Cho, and 57/76
Physionet targets.

The locked decision precedence is C84-E, C84-D, C84-A, C84-B, C84-C. A/B
require the same fixed method across all datasets and all stability gates.
Label-frontier tags C84-L1--L4 use raw budget curves, larger-budget closure,
level-specific qualification, method-free budget identity, and target/level
stability; no isotonic repair is permitted.

## Complete result freeze

The production writer covers all required outputs, including standalone
coverage, selected-regime, and catastrophic-failure tables. Sixteen Stage-C
CSV schemas, `C84S_RESULT.json`, and the artifact manifest are validated in a
temporary directory and atomically published only after referential integrity,
row counts, schemas, hashes, taxonomy, and protected counters pass.

An injected partial-write failure leaves no final result root. A real C84S
failure after label access cannot publish a partial scientific identity.

## Synthetic calibration

All S0--S20 scenarios ran through production public entrypoints and passed
21/21. Coverage includes:

```text
C84-A/B/C/D outcomes;
C84-L1/L2/L3/L4 outcomes;
same-method cross-dataset identity;
level, panel/seed, and LOTO heterogeneity;
measurement/top-k non-substitution for Q1;
target-row and Monte Carlo pseudoreplication rejection;
evaluation-before-selection and split-overlap rejection;
trial-order drift rejection;
exact 9,621-row label alignment;
atomic publication failure.
```

The complete S0 fixture exercised all 18,608 method-context rows and every
registered production table. Synthetic calibration accessed zero real labels
and zero real selector values.

## Verification summary

```text
focused:  55 passed
C65:      813 passed, 1 skipped, 3 deselected
C23:    1,224 passed, 1 skipped, 3 deselected
full:   2,148 passed, 1 skipped, 3 deselected

accepted stderr bytes: 0 for every final suite
synthetic production scenarios: 21 / 21 PASS
static isolation checks:         16 / 16 PASS
external artifact hash replay: 3,888 / 3,888 PASS
runtime implementation files:    12 / 12 bound by SHA-256 and Git blob
readiness tables:                 21 / 21 bound by SHA-256
```

The one cumulative skip is the finalized C78F lifecycle test. The three
deselections are the registered historical C79 authorization-state checks.
Full regression history, including failed and deliberately canceled initial
attempts, is in `C84SL_REGRESSION_VERIFICATION.md`.

## Authorization boundary

No C84S authorization record exists at readiness. The old C84F authorization
does not carry forward. Future execution requires a new direct PI statement:

```text
授权 C84S
```

The server must bind that statement to the exact V2 lock above. C84SL does not
authorize C85, manuscript changes, new methods, new hyperparameters, or any
other experiment.

