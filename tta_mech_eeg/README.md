# TTA-MECH-EEG

Mechanistic audit of target-unlabeled EEG adaptation; benchmark and
explanation, not a new method.

TTA-MECH-EEG studies why existing target-unlabeled adaptation baselines such as
TTA-Control, matched-CORAL, SPDIM, and T3A can improve target EEG performance,
when those gains reproduce, and which observable mechanisms carry them.

This project is not TALOS_01, CEDAR_02, CITA-CMI rescue, CutClean-for-EEG,
ACAR, TOS, OACI, h2cmi, FSR, or any other method rescue line.

## Current Status

```text
TTA_MECH_00 design package: drafted
TTA_MECH_00A artifact-inventory / replay-harness preflight: PASS
TTA_MECH_01 real existing-baseline replay / mechanism audit: MECHANISM_INFORMATIVE_PASS
new method: none
P1 / P2 training claim: none
source-free deployment claim: none
```

## Active Taxonomy

```text
TM1_existing_baseline_replay
TM2_entropy_balance_vs_geometry_decomposition
TM3_source_replay_ablation
TM4_normalization_batchnorm_audit
TM5_accuracy_vs_calibration_separation
TM6_target_label_quarantine_required
TM7_no_new_method_claim
```

## Inactive / Forbidden Taxonomy

```text
TM8_new_adapter
TM9_cmi_control
TM10_pruning_mask_surgery
TM11_safety_gate_router
TM12_target_informed_selection
TM13_source_free_deployment_claim
```

## Reports

```text
tta_mech_eeg/reports/TTA_MECH_00_PROJECT_CHARTER.md
tta_mech_eeg/reports/TTA_MECH_00_BENCHMARK_PROTOCOL.md
tta_mech_eeg/reports/TTA_MECH_00_ACCEPTANCE_CRITERIA.md
tta_mech_eeg/reports/TTA_MECH_00_PM_BOUNDARIES.md
tta_mech_eeg/reports/TTA_MECH_00_FAILURE_PRIORS.md
tta_mech_eeg/reports/TTA_MECH_00A_PREFLIGHT_READOUT.md
tta_mech_eeg/reports/TTA_MECH_01_REAL_REPLAY_READOUT.md
```
