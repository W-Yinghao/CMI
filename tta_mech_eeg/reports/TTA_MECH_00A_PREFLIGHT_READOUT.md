TTA_MECH_00A - Artifact Inventory and Replay-Harness Preflight Readout

This is an artifact-inventory and replay-harness preflight.
No real EEG replay was run.
No target metrics were computed.
No baseline was selected.
No new method was introduced.
TTA-MECH remains a mechanism-audit project only.

Status: PASS.

Scope

```text
design_baseline_commit: cccd8c1
phase: TTA_MECH_00A_artifact_inventory_replay_harness_preflight
real_eeg_replay_run: false
target_metrics_computed: false
baseline_selected: false
new_method_introduced: false
```

Runtime artifact root

```text
results/tta_mech/tta_mech00a_preflight/
```

Primary hashes

```text
baseline_registry_hash: 0d8a00cdb2d0bf810a20c58056323c21c1fde8807fb12e65ebc9ff8334748da7
artifact_inventory_hash: 6323097829f2d3277275f392ea33329a4197e7032969bb3ebf87d1ad2e090cb2
preflight_payload_hash: e93238a182b02f953da9123e7219610fb82e50e361645b263c6adf7079472b32
```

Baseline registry

Frozen baselines:

```text
ERM_NO_ADAPT
TTA_CONTROL_REPLAY
MATCHED_CORAL
SPDIM
T3A
```

Forbidden baseline families remain absent:

```text
TALOS
CEDAR
CITA_CMI
CMI_REGULARIZED
LOW_RANK_ADAPTER
NEW_METHOD
SAFETY_GATE
```

Artifact inventory

Inventory summary:

```text
total_records: 18
cedar01f_feature_records: 18
available_records: 18
missing_records: 0
rejected_records: 0
```

The inventory validates CEDAR_01F handoff availability and per-artifact file
hashes, and inspects npz key names for availability fields such as z, y,
domain, and groups. It does not run baseline replay and does not compute target
metrics.

Audit-axis schema

The preflight freezes schema support for:

```text
entropy_confidence
balance_prior
geometry
source_replay
normalization_batchnorm
calibration
```

Red-team checks

```text
baseline_universe_freeze: PASS
target_label_quarantine_contract: PASS
replay_determinism: PASS
no_new_method_guard: PASS
```

Baseline universe freeze checks:

```text
allowed_baselines_exact
entry_names_exact
forbidden_baselines_absent
entries_existing_label_free_allowed
runtime_addition_disabled
baseline_registry_hash_recomputable
```

Target-label quarantine contract:

```text
callable: adapt_or_replay
required parameters: source_state, target_x, baseline
forbidden parameters absent: target_y, y_target, target_metric, target_selected_variant
```

Toy replay determinism:

```text
toy_output_hash_identical
baseline_identical
toy_output_hash: 202d7d85324ad01605a54e46f111610dfdf8c42f5e2e9d6c08bcd71226659f01
```

No-new-method guard:

```text
no_forbidden_active_method_terms
new_method_claim_false
```

Forbidden active method terms checked:

```text
adapter
learned_operator
mask
prune
surgery
cmi_control
safety_gate
router
```

Boundary statement

TTA_MECH_00A is not TTA_MECH_01. It does not run BNCI replay, does not compute
bAcc / NLL / ECE tables, does not select a best baseline, does not add a
baseline, and does not implement a new adapter or objective.

Next gate

Only after PM accepts this preflight may TTA_MECH_01 be requested. A future
TTA_MECH_01 would still be an existing-baseline replay / mechanism audit, not a
new method or P1/P2 training request.
