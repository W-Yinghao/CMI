TALOS_00 - Project Charter

Status: design-only charter. TALOS_00 is approved for protocol design, not yet
for execution. This file does not authorize P1 training, P2 streaming,
structured pruning, CMI control, or any target-informed selection.

Project name:

```text
TALOS-EEG
Target-unlabeled Adaptation with Low-dimensional Operator Shrinkage for EEG
```

One-line objective:

```text
Use a small target-unlabeled feature/logit adapter to explain and strengthen the
non-CMI adaptation lever observed in EEG TTA, without pruning, source-only
safety gates, or CMI control.
```

Hard boundary inherited from CEDAR

CEDAR_01N is frozen as a real EEG negative closeout:

```text
source-only frozen-latent leakage evidence did not produce actionable deletion
0 / 54 CEDAR_01 candidates accepted
P1 channel pruning denied
P2 TTA preconditioner denied
deployable mask forbidden
generalization / safety claim forbidden
```

Therefore TALOS is not:

```text
latent mask retry
channel pruning retry
graph surgery retry
CutClean-for-EEG retry
CMI control retry
source-only safety gate / router retry
```

Scientific motivation

The project portfolio now has one robust positive signal and several negative
boundaries:

```text
CMI control: negative
source-only deletion / surgery: negative
harm routing / safety gate: negative or not certified
target-unlabeled adaptation: repeatedly positive, but not explained by CMI
```

TALOS asks which low-dimensional non-CMI operator carries that target-unlabeled
adaptation gain.

Core hypothesis

Held-out EEG subject mismatch can often be reduced by a small constrained
target-unlabeled operator that adjusts confidence, prior, feature centering, or
low-dimensional feature geometry. This does not require deleting subject
information, updating the encoder, updating the classifier head, or claiming a
safety gate.

Hypothesis layers:

```text
H1: The TTA gain is largely low-dimensional logit / feature geometry correction.
H2: Serialized source statistics should be enough for the adapter objective.
H3: TALOS should use trust-region constraints, not label-free action gates.
```

TALOS_00 scope

TALOS_00 is a frozen-feature replay stage using already-produced
CEDAR_01F artifacts:

```text
dataset: BNCI2014_001
backbones: EEGNetMini, EEGConformerMini
seed: 0
folds: full LOSO, 9 folds per backbone
feature source: results/cedar/feature_supply/cedar01f_bnci2014_001_seed0
```

TALOS_00 may read:

```text
source_train features and labels
source_audit features and labels for source-only diagnostics
target_audit features without labels during adapter fitting
target_audit labels only for final metrics
CEDAR_01F handoff manifest and per-artifact hashes
```

TALOS_00 must not read target labels for:

```text
adapter fitting
variant choice
hyperparameter choice
early stopping
trust-region tuning
collapse rescue
failure taxonomy rescue
```

Initial variant universe

The TALOS_00 variant universe is frozen before execution:

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

Definitions:

```text
TALOS-L:  logit bias + temperature only
TALOS-D:  diagonal feature affine only
TALOS-LD: logit bias + temperature + diagonal feature affine
```

TALOS-LR and TALOS-full are not part of the first execution unless PM approves
a protocol amendment before any run. Full encoder adaptation is explicitly out
of scope.

Primary metrics

```text
target balanced accuracy
target NLL
target ECE
worst-fold target balanced accuracy
target entropy / label-balance collapse guards
adapter norm
source-audit diagnostic metrics
```

Target metrics are final evaluation only and cannot select the TALOS variant.

Retained assets from prior projects

```text
CEDAR feature schema / manifest validation
target-label quarantine discipline
grouped EEG split requirements
negative-result language for measurement-to-control gap
CITA / TTA-Control non-CMI positive signal
```

Non-goals

```text
No SOTA claim.
No CMI mechanism claim.
No privacy or safety claim.
No target-generalization certificate.
No action gate.
No deployment claim.
No pruning, masks, deletion, or surgery.
```

Exit from TALOS_00

TALOS_00 is only a P0 decision experiment. If it passes, PM may consider
TALOS_01 source-free serialized-state adaptation. If it fails, TALOS closes as a
diagnostic replay result, not as a method pipeline.
