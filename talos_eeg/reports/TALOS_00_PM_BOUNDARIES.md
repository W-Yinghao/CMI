TALOS_00 - PM Boundaries

Status: design-only boundary record. This file converts the PM decision into
explicit project constraints for TALOS_00. It does not authorize execution.

Decision

```text
Start TALOS-EEG as a new project.
Do not continue CEDAR as CEDAR_02.
Treat CEDAR_01N as a hard negative boundary.
```

CEDAR boundary absorbed by TALOS

```text
source-only frozen-latent leakage evidence did not produce actionable deletion
real EEG CEDAR_01 result: 0 / 54 candidates accepted
CEDAR source-only deletion / pruning / surgery route is closed negative
```

Forbidden continuations

```text
latent mask retry
channel pruning retry
graph surgery retry
CutClean-for-EEG retry
CMI control retry
source-only safety gate retry
source-only router retry
CEDAR_02 relabeling of the same method path
```

Allowed TALOS_00 design target

```text
feature-space adapter replay
BNCI2014_001
EEGNetMini + EEGConformerMini
seed0
existing CEDAR_01F handoff features
low-dimensional target-unlabeled feature/logit operators
calibration-aware metrics
strict target-label quarantine
```

Not authorized in TALOS_00

```text
new P1 training
new GPU model training
full encoder adaptation
target-label hyperparameter tuning
target-informed variant selection
source-free deployment claim
privacy claim
safety claim
SOTA claim
clinical transfer claim
```

Source-free boundary

TALOS_00 may use the existing CEDAR_01F source rows to derive replay source
statistics because this stage is a frozen-feature design/replay stage. A
source-free deployment claim is not allowed unless a future approved runner
passes a source-state guard showing that the adapter reads serialized source
statistics and not source examples.

P1 / P2 boundary

```text
TALOS_01: not approved
P1 source-free serialized-state adaptation: not approved
P2 streaming / clinical transfer: not approved
```

The only approved next deliverable is the TALOS_00 design package:

```text
TALOS_00_PROJECT_CHARTER.md
TALOS_00_FROZEN_FEATURE_PROTOCOL.md
TALOS_00_ACCEPTANCE_CRITERIA.md
TALOS_00_PM_BOUNDARIES.md
```
