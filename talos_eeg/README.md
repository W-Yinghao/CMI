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
TALOS_00 execution: not run
TALOS_00B real frozen-feature replay: not approved / not run
P1 source-free serialized-state training: not approved
P2 streaming / clinical transfer: not approved
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
No source-free deployment claim until source-state guard passes.
No safety-gate claim.
```

## Reports

```text
talos_eeg/reports/TALOS_00_PROJECT_CHARTER.md
talos_eeg/reports/TALOS_00_FROZEN_FEATURE_PROTOCOL.md
talos_eeg/reports/TALOS_00_ACCEPTANCE_CRITERIA.md
talos_eeg/reports/TALOS_00_PM_BOUNDARIES.md
talos_eeg/reports/TALOS_00A_PREFLIGHT_READOUT.md
```
