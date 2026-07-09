TALOS_00A - Adapter Implementation and Red-Team Preflight Readout

Status: PASS.

This is an infrastructure gate, not a scientific readout. The run used
synthetic smoke data only and did not read or evaluate real CEDAR_01F
BNCI2014_001 feature artifacts.

PM boundary

```text
TALOS_00 design package: PASS at 96894a7
TALOS_00A implementation/red-team preflight: PASS
TALOS_00B real frozen-feature replay: NOT_RUN / not approved by this readout
TALOS_P1: BLOCKED
TALOS_P2: BLOCKED
CEDAR continuation: FORBIDDEN
CMI / pruning / mask / surgery / safety-gate rescue: FORBIDDEN
source-free deployment claim: false
```

Implemented scope

```text
source state schema for P0 frozen-feature replay
TALOS-L  logit bias + temperature
TALOS-D  diagonal feature affine
TALOS-LD diagonal feature affine + logit bias + temperature
trust-region hard bounds
collapse guards
target-label quarantine check
adapter determinism check
variant freeze check
synthetic preflight runner
```

Not implemented in TALOS_00A

```text
TALOS-LR
TALOS-full
class-conditional geometry loss
prior uncertainty beyond a simple source-prior interval
online streaming
clinical PD / SCZ transfer
CMI loss
CEDAR mask / pruning / surgery
safety gate / harm router
```

Runtime artifact root

```text
results/talos/talos00a_preflight
```

Primary run manifest

```text
status: PASS
phase: TALOS_00A_adapter_implementation_red_team_preflight
design_baseline_commit: 96894a7
implementation_scope: synthetic_smoke_only_no_real_eeg_readout
real_eeg_readout_run: false
scientific_readout: false
talos00b_real_replay_approved: false
source_free_deployment_claim: false
preflight_payload_hash: 14a76cff7367fced4fa46c8915ccdb2240ae81857629a1265c1d5aa06fb34727
```

Variant universe

Frozen allowed variants:

```text
ERM
TTA_CONTROL_REPLAY
TALOS_L
TALOS_D
TALOS_LD
```

Forbidden variants recorded by the preflight:

```text
TALOS_LR
TALOS_full
CMI
CEDAR_mask
safety_gate
```

Variant freeze result:

```text
passed: true
variant_universe_hash: 328482d68fb458c23c71a900e8f16aae1d8fdd7553052dc29a856d33c45b50d4
checks:
  allowed_variants_exact
  runtime_addition_disabled
  forbidden_variants_recorded
  target_labels_final_metrics_only
  variant_universe_hash_recomputable
warnings: []
```

Target-label quarantine

Scenarios:

```text
true_y_final_only
target_y_removed
target_y_permuted
```

Result:

```text
passed: true
checks:
  adapter_parameters_identical
  adapter_predictions_identical
  adapter_probability_hashes_identical
  variant_ranking_identical
  reported_variant_identical
  pre_final_metric_hashes_identical
  removed_y_has_no_final_metrics
  only_final_metrics_change_under_permutation
warnings: []
```

Interpretation: target labels do not affect adapter parameters, predicted
labels, probability outputs, pre-final metrics, variant ranking, or the reported
variant. Target labels only affect synthetic final metrics.

Adapter determinism

```text
passed: true
checks:
  scenario_match
  variant_ranking_identical
  reported_variant_identical
  adapter_prediction_probability_hashes_identical
warnings: []
```

Source-state schema

```text
passed: true
source_state_mode: constructed_from_frozen_source_features_for_P0_replay
source_free_deployment_claim: false
source_state_hash: c371507e86c8b2afe08b51f8650ffbdcbddd4d31d8d49186edb1cbdf79335a65
checks:
  source_state_mode_p0_replay
  source_free_deployment_claim_false
  finite_vector_fields
  array_shapes
  source_prior_simplex
  positive_scale_fields
warnings: []
```

Trust-region / collapse status

All frozen variants stayed inside the predeclared trust-region bounds:

```text
tau_log_t: 0.25
tau_beta: 1.50
tau_a: 0.35
tau_c: 1.50
boundary_hits: none
```

All frozen variants passed collapse guards on synthetic target features:

```text
entropy_non_degenerate
not_single_class
warnings: []
```

Tests

```text
PYTHONPATH=. pytest -q talos_eeg/tests
5 passed

python -m compileall -q talos_eeg
PASS
```

Scientific boundary

This readout does not evaluate TALOS on real EEG. It does not report a
BNCI2014_001 target bAcc table, does not choose a final method, does not request
P1, and does not create a source-free deployment, safety, privacy, SOTA, or
target-generalization claim.

Next gate

Only after PM accepts this TALOS_00A preflight should TALOS_00B be considered.
TALOS_00B would be the first approved real frozen-feature adapter replay over
the 18 CEDAR_01F BNCI2014_001 handoff artifacts.
