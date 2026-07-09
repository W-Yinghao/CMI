TTA_MECH_01 - Real Existing-Baseline Replay Readout

This is a mechanism-audit replay of existing baselines.
No new method was introduced.
No baseline was selected for deployment.
Target labels were final-metric-only.
No CMI, pruning, mask, surgery, safety-gate, or router was active.

Status: MECHANISM_INFORMATIVE_PASS.

This is not a method pass/fail result, not a deployment decision, and not a
P1/P2 training claim. The replay uses only the fixed TTA-MECH baseline universe
over the compliant CEDAR_01F BNCI2014_001 frozen-feature handoff.

Scope

```text
preflight_baseline_commit: 3c4d0fe
phase: TTA_MECH_01_existing_baseline_real_replay_mechanism_audit
dataset: BNCI2014_001
backbones: EEGNetMini, EEGConformerMini
seed: 0
folds: 9 per backbone, 18 artifacts total
real_eeg_replay_run: true
target_metrics_final_only: true
baseline_selected_for_deployment: false
new_method_introduced: false
p1_p2_training: false
cmi_pruning_mask_surgery_safety_router_active: false
```

Runtime artifact root

```text
results/tta_mech/tta_mech01_bnci2014_001_seed0/
```

Primary hashes

```text
baseline_registry_hash: 0d8a00cdb2d0bf810a20c58056323c21c1fde8807fb12e65ebc9ff8334748da7
artifact_inventory_hash: 6323097829f2d3277275f392ea33329a4197e7032969bb3ebf87d1ad2e090cb2
tta_mech01_payload_hash: f8e99c99b9b61f66f0011da60620220021f861b4b409b8e46d6a781cc131a816
```

Output files

```text
run_manifest.json
baseline_registry_hash.txt
artifact_inventory_hash.txt
per_fold_metrics.csv
audit_axes_table.csv
aggregate_baseline_table.csv
mechanism_summary.json
target_label_quarantine.json
replay_determinism.json
red_team.json
no_new_method_guard.json
artifact_validation.json
tta_mech01_summary.json
```

Replay coverage

```text
artifact_validation: PASS
loaded_artifacts: 18
per_fold_metric_rows: 90
audit_axis_rows: 90
aggregate_rows: 10
baselines_replayed: ERM_NO_ADAPT, TTA_CONTROL_REPLAY, MATCHED_CORAL, SPDIM, T3A
source_replay_axis: NOT_AVAILABLE_IN_THIS_REPLAY
BN_axis: NOT_TESTED_IN_FROZEN_FEATURE_REPLAY
```

Red-team checks

```text
target_label_noninterference: PASS
baseline_universe_hash: PASS
artifact_inventory_hash: PASS
artifact_hash_validation: PASS
replay_determinism: PASS
no_new_method_guard: PASS
baseline_universe_freeze: PASS
red_team_failures: 0
```

Target-label quarantine details

```text
premetric_outputs_identical
baseline_outputs_identical_before_final_metric
audit_premetric_artifacts_identical
removed_target_y_has_no_final_metrics
permuted_target_y_differs_only_after_final_metric
final_metric_hash_differences_vs_permuted_y: 90
```

Aggregate mechanism table

| Backbone | Baseline | mean bAcc | delta bAcc vs ERM | mean NLL | delta NLL vs ERM | dominant observable axis |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| EEGConformerMini | ERM_NO_ADAPT | 0.4136 | 0.0000 | 5.1844 | 0.0000 | reference_no_adaptation |
| EEGConformerMini | TTA_CONTROL_REPLAY | 0.4136 | 0.0000 | 4.8540 | -0.3304 | calibration_temperature_effect |
| EEGConformerMini | MATCHED_CORAL | 0.4246 | 0.0110 | 6.3395 | 1.1550 | geometry_alignment_with_accuracy_gain |
| EEGConformerMini | SPDIM | 0.4292 | 0.0156 | 4.4393 | -0.7451 | geometry_no_clear_alignment |
| EEGConformerMini | T3A | 0.3964 | -0.0172 | 5.2772 | 0.0928 | classifier_template_adjustment_without_accuracy_gain |
| EEGNetMini | ERM_NO_ADAPT | 0.4005 | 0.0000 | 2.4672 | 0.0000 | reference_no_adaptation |
| EEGNetMini | TTA_CONTROL_REPLAY | 0.4005 | 0.0000 | 2.4182 | -0.0490 | calibration_temperature_effect |
| EEGNetMini | MATCHED_CORAL | 0.4583 | 0.0579 | 1.9578 | -0.5094 | geometry_alignment_with_accuracy_gain |
| EEGNetMini | SPDIM | 0.4527 | 0.0523 | 1.6384 | -0.8288 | geometry_no_clear_alignment |
| EEGNetMini | T3A | 0.4142 | 0.0137 | 3.3750 | 0.9078 | classifier_template_adjustment_with_accuracy_gain |

Mechanism observations

TTA_CONTROL_REPLAY changes calibration without changing predicted labels in
this frozen-feature replay. The bAcc delta is 0.0000 on both backbones, while
NLL decreases on both EEGConformerMini and EEGNetMini.

MATCHED_CORAL shows a geometry-associated bAcc gain on both backbones. On
EEGNetMini, the gain is accompanied by lower NLL and lower ECE. On
EEGConformerMini, bAcc improves while NLL and ECE worsen, so the readout marks a
bAcc-NLL divergence at aggregate level.

SPDIM shows bAcc gains on both backbones and lower NLL on both backbones, but
the CORAL-distance delta does not identify a clear covariance-alignment
mechanism. The observable mechanism is recentering, with the source replay and
BN axes unavailable in this frozen-feature replay.

T3A is backbone-dependent in this replay. EEGConformerMini loses bAcc with a
small NLL increase. EEGNetMini gains bAcc but worsens NLL, which is a
classification/calibration divergence rather than a deployment signal.

Boundary statement

The result is mechanism-informative because every frozen baseline is replayed,
the audit axes are populated for all fold x backbone x baseline rows, red-team
checks are clean, and each backbone/baseline pair receives a dominant observable
axis. It does not select a best baseline, does not introduce a new method, and
does not justify deployment.
