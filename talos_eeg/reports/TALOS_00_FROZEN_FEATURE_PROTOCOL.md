TALOS_00 - Frozen Feature Protocol

Status: design protocol only. This document specifies the first allowed TALOS
experiment but does not run it.

Objective

Run a target-unlabeled low-dimensional adapter replay on existing frozen
features to test whether non-CMI TTA gains can be explained by small logit or
feature-space operators.

Approved input feature set

```text
feature root:
  results/cedar/feature_supply/cedar01f_bnci2014_001_seed0

handoff manifest:
  results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/CEDAR_01F_HANDOFF_MANIFEST.json

dataset:
  BNCI2014_001

backbones:
  EEGNetMini
  EEGConformerMini

seed:
  0

folds:
  9 LOSO target folds per backbone
```

The runner must validate the CEDAR_01F handoff manifest and hard-fail on any
feature hash mismatch before fitting an adapter.

Feature roles

Each feature artifact contains:

```text
source_train
source_audit
target_audit
```

Allowed use:

```text
source_train:
  source statistics, prototypes, covariance / diagonal covariance, source
  readout replay, optional source-side diagnostics

source_audit:
  source-only sanity checks and collapse diagnostics

target_audit features:
  target-unlabeled adapter fitting

target_audit labels:
  final metrics only
```

Forbidden use of target labels:

```text
adapter gradients / fitting objective
variant selection
hyperparameter selection
early stopping
trust-region tuning
collapse rescue
reranking
acceptance rescue
```

Source state for TALOS_00

Because CEDAR_01F artifacts are frozen features and do not include classifier
head weights, TALOS_00 must use feature-space replay readouts and serialized
source statistics derived only from source rows:

```text
class prototypes:                 mu_y
diagonal or shrinkage covariance: Sigma_y
source label prior:               pi_s
source subject prior intervals:   Pi_s
source feature normalization:     feature_norm_stats
source calibration temperature:   T_s, if fit on source rows only
```

If a future execution uses frozen model logits or classifier weights, they must
be recorded as part of the source state before target labels are visible.

Adapter family

TALOS_00 is limited to low-dimensional feature/logit replay operators:

```text
z' = A z + c
logits' = replay_head(z') + beta
p' = softmax(logits' / T)
```

Allowed operators in TALOS_00:

```text
TALOS-L:
  beta logit bias
  T temperature

TALOS-D:
  A diagonal feature scale
  c feature recentering bias

TALOS-LD:
  diagonal A
  c
  beta
  T
```

Initialization:

```text
A = I
c = 0
beta = 0
T = 1
```

Trust region:

```text
||A - I||^2
||c||^2
||beta||^2
(log T)^2
```

Objective components

Target-unlabeled objective may include only:

```text
entropy / confidence term
prior uncertainty penalty against source subject prior set Pi_s
soft class-conditional geometry term against source prototypes / covariance
trust-region regularization
```

Explicitly forbidden:

```text
CMI term
privacy head
mask loss
deletion objective
target supervised loss
target-label balance loss using true labels
source replay examples in source-free mode
encoder update
classifier-head update
```

Baselines

TALOS_00 must compare against:

```text
ERM-no-adapt
TTA-Control feature/logit replay
matched-CORAL
SPDIM-style recentering
T3A
TALOS-L
TALOS-D
TALOS-LD
```

CITA-CMI is not an active baseline. It may appear only as a historical
report-only reference.

Hyperparameter freeze

Before any execution, the runner must serialize:

```text
variant universe
adapter parameterization
trust-region bounds
objective weights
optimization steps
random seeds
metric list
collapse thresholds
```

Target labels must not be used to choose or amend these values.

Required red-team checks

```text
target-label noninterference:
  removing, shuffling, or replacing target labels does not change adapter
  parameters or adapter output hashes

adapter determinism:
  same source state + target X gives identical adapter output hash

source-free guard:
  source-free mode reads serialized source stats, not source examples

variant universe freeze:
  no variant, rank, or transform is added after observing metrics

collapse guard:
  target predictions are not single-class and entropy is non-degenerate

adapter norm bound:
  trust-region norms remain within predeclared bounds

no target-informed variant selection:
  final target metrics do not decide which variant is reported as primary
```

Output layout for a future TALOS_00 run

```text
results/talos/talos00_bnci2014_001_seed0/
  run_manifest.json
  frozen_handoff_hash.txt
  source_state_manifest.json
  variant_universe.json
  adapter_table.csv
  adapter_table.hash
  red_team.json
  target_metrics_FINAL_ONLY.json
  collapse_guards.json
```

No `.npz`, model checkpoint, deployable adapter, or source example archive
should be committed.

Execution status

This protocol has not been executed. Running TALOS_00 requires a separate PM
approval after this design package is accepted.
