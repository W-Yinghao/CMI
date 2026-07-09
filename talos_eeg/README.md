# TALOS-EEG

Target-unlabeled Adaptation with Low-dimensional Operator Shrinkage for EEG.

TALOS is a new project line, not a continuation of CEDAR. CEDAR_01N is a hard
boundary: source-only frozen-latent leakage evidence did not produce actionable
deletion on real EEG features. TALOS does not retry latent masks, channel
pruning, graph surgery, CutClean-for-EEG, CMI control, source-only safety gates,
or routers.

## Scope

TALOS studies a narrower non-CMI question:

```text
Can a low-dimensional target-unlabeled operator explain and preserve the robust
EEG TTA gain without pruning, CMI, target-label selection, or a safety-gate
claim, while reducing source dependence to serialized state before any
source-free claim?
```

The initial phase is TALOS_00: frozen-feature adapter replay on the existing
CEDAR_01F BNCI2014_001 feature artifacts.

## Current Status

```text
TALOS_00_PROJECT_CHARTER: designed
TALOS_00_FROZEN_FEATURE_PROTOCOL: designed
TALOS_00_ACCEPTANCE_CRITERIA: designed
TALOS_00_PM_BOUNDARIES: designed
TALOS_00_DESIGN_PACKAGE: PASS at 96894a7
TALOS_00A_IMPLEMENTATION_PREFLIGHT: PASS
TALOS_00 scientific gate: FAIL at TALOS_00B
TALOS_00B real frozen-feature replay: COMPLETE_NEGATIVE / diagnostic-only
TALOS_00N negative closeout: complete
TALOS low-degree-of-freedom frozen-feature route: CLOSED_NEGATIVE
P1 source-free serialized-state training: denied from TALOS_00B
P2 streaming / clinical transfer: denied
```

## Hard Boundaries

```text
No CEDAR_02 retry.
No latent mask retry.
No channel pruning retry.
No graph surgery retry.
No CMI control retry.
No CutClean-for-EEG pruning pipeline.
No target labels except final metrics.
No target-informed variant selection.
No TALOS-LR or TALOS-full rescue from TALOS_00B.
No trust-region relaxation rescue from TALOS_00B.
No source-free deployment claim from TALOS_00.
No safety-gate claim.
```

## Reports

```text
talos_eeg/reports/TALOS_00_PROJECT_CHARTER.md
talos_eeg/reports/TALOS_00_FROZEN_FEATURE_PROTOCOL.md
talos_eeg/reports/TALOS_00_ACCEPTANCE_CRITERIA.md
talos_eeg/reports/TALOS_00_PM_BOUNDARIES.md
talos_eeg/reports/TALOS_00A_PREFLIGHT_READOUT.md
talos_eeg/reports/TALOS_00B_REAL_FROZEN_FEATURE_READOUT.md
talos_eeg/reports/TALOS_00N_NEGATIVE_CLOSEOUT.md
talos_eeg/reports/TALOS_00N_FAILURE_TAXONOMY.md
talos_eeg/reports/TALOS_00N_PM_DECISION.md
```
