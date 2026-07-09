TTA_MECH_01S - PM Decision Request

TTA_MECH_01S is complete as a read-only mechanism synthesis and is submitted for
PM review. It does not start TTA_MECH_02.

Boundary lock

```text
TTA_MECH_01_REAL_REPLAY_BASELINE: db57f6d
NEW_REPLAY_RUN: false
NEW_METHOD_CLAIM: false
BASELINE_DEPLOYMENT_SELECTION: false
P1/P2_TRAINING_REQUEST: false
CEDAR_RESCUE: false
TALOS_RESCUE: false
CMI_CONTROL_RESCUE: false
CUT_CLEAN_PRUNING_FRAMING: false
```

Files for review

```text
tta_mech_eeg/reports/TTA_MECH_01S_MECHANISM_SYNTHESIS.md
tta_mech_eeg/reports/TTA_MECH_01S_BASELINE_MECHANISM_MATRIX.md
results/tta_mech/tta_mech01_bnci2014_001_seed0/mechanism_matrix.csv
results/tta_mech/tta_mech01_bnci2014_001_seed0/mechanism_synthesis.json
```

Decision summary

TTA_MECH_01S separates accuracy and calibration effects. The main readout is:

```text
TTA_CONTROL_REPLAY: calibration-only, no bAcc gain
MATCHED_CORAL: geometry-alignment signal, calibration relation backbone-specific
SPDIM: recentering / geometry signal, accuracy and calibration aligned
T3A: backbone-specific classifier-template behavior
source replay axis: unavailable
BN / normalization axis: unavailable in frozen-feature replay
```

Recommendation for PM

```text
Recommend TTA_MECH_02B_NORMALIZATION_BN_AUDIT
```

This is only a recommendation. It does not authorize TTA_MECH_02. The rationale
is that the clearest remaining mechanism gap is normalization / BN, because
geometry and recentering signals appear in MATCHED_CORAL and SPDIM while
TTA_MECH_01 frozen features cannot test BN or normalization directly.

Alternatives not recommended now

```text
TTA_MECH_02A_SOURCE_REPLAY_ABLATION
CLOSE_AS_MECHANISM_BENCHMARK_ONLY
```

PM decision requested

Choose one:

```text
OPEN_TTA_MECH_02B_NORMALIZATION_BN_AUDIT
OPEN_TTA_MECH_02A_SOURCE_REPLAY_ABLATION
CLOSE_AS_MECHANISM_BENCHMARK_ONLY
REQUEST_01S_REVISION
```
