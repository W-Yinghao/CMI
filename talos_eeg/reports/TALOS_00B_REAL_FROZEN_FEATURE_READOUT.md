TALOS_00B - Real Frozen-Feature Adapter Replay Readout

This is a real frozen-feature replay experiment, not P1 training.
Target labels are final-metric-only.
No source-free deployment claim is made.
No CMI, pruning, mask, surgery, or safety-gate mechanism is active.

Status: FAIL / diagnostic-only.

TALOS_00B ran the approved real replay gate over the CEDAR_01F BNCI2014_001
handoff features. The infrastructure and red-team checks passed, but the
scientific gate did not pass: EEGConformerMini did not reach the required
+0.020 bAcc improvement, and the largest EEGNetMini gains came from
boundary-hit adapters.

Run Scope

```text
phase: TALOS_00B_real_frozen_feature_adapter_replay
preflight_baseline_commit: b8dbb70
dataset: BNCI2014_001
backbones: EEGNetMini, EEGConformerMini
seed: 0
folds: 9 LOSO folds per backbone
feature_artifacts_loaded: 18
real_frozen_feature_replay: true
p1_training: false
source_free_deployment_claim: false
talos00b_payload_hash: 68ae5140dd5131df69bcb8d17d6af4b973d3b56129480eeb1e2e31f81d3da199
```

Runtime artifacts:

```text
results/talos/talos00b_bnci2014_001_seed0/
  run_manifest.json
  feature_handoff_validation.json
  variant_table.csv
  per_fold_metrics.csv
  red_team.json
  target_label_quarantine.json
  adapter_norms.json
  collapse_guards.json
  calibration_metrics.json
  source_state_summary.json
  scientific_gate.json
  talos00b_summary.json
```

Handoff Immutability

```text
cedar01f_handoff_hash: 03d97199352892bf39afeed3ce826aa185735a766351b82aa2083587621d289c
cedar01f_canonical_payload_hash: e5f76423c673db9f7192c327812942385ed81bfe3949d4a7d30c5e86b70e9642
feature_artifacts_expected: 18
feature_artifacts_loaded: 18
per_artifact_hash_check: PASS
```

Red-Team Result

Red-team failures: 0.

Red-team warnings: 0.

```text
handoff_immutability: PASS
target_label_quarantine: PASS
adapter_determinism: PASS
variant_freeze: PASS
```

Target-label quarantine was repeated on the real BNCI2014_001 features:

```text
true_y_final_only
target_y_removed
target_y_permuted
```

Checks passed:

```text
adapter_parameters_identical
adapter_predictions_identical
adapter_probability_hashes_identical
variant_ranking_identical
reported_variant_identical
pre_final_metric_hashes_identical
removed_y_has_no_final_metrics
only_final_metrics_change_under_permutation
```

Variant Universe

Allowed variants:

```text
ERM_NO_ADAPT
TTA_CONTROL_REPLAY
TALOS_L
TALOS_D
TALOS_LD
```

Forbidden mechanisms remained absent:

```text
TALOS_LR
TALOS_FULL
rank-r affine
geometry-loss full variant
CMI
CEDAR mask
pruning
surgery
safety gate / harm router
```

Aggregate Results

Balanced accuracy, NLL, and ECE are target final metrics only. They were not
used to fit adapters or select a deployable variant.

```text
EEGConformerMini
  ERM_NO_ADAPT        bAcc 0.4136  NLL 5.1844  ECE 0.5090  boundary 0/9
  TTA_CONTROL_REPLAY  bAcc 0.4136  NLL 4.8540  ECE 0.5015  boundary 0/9
  TALOS_L             bAcc 0.4157  NLL 4.7819  ECE 0.4965  boundary 0/9
  TALOS_D             bAcc 0.4180  NLL 5.8497  ECE 0.5201  boundary 9/9
  TALOS_LD            bAcc 0.4188  NLL 5.4760  ECE 0.5111  boundary 9/9

EEGNetMini
  ERM_NO_ADAPT        bAcc 0.4005  NLL 2.4672  ECE 0.3441  boundary 0/9
  TTA_CONTROL_REPLAY  bAcc 0.4005  NLL 2.4182  ECE 0.3416  boundary 0/9
  TALOS_L             bAcc 0.4209  NLL 2.1447  ECE 0.3027  boundary 1/9
  TALOS_D             bAcc 0.4579  NLL 1.7499  ECE 0.2474  boundary 9/9
  TALOS_LD            bAcc 0.4581  NLL 1.7503  ECE 0.2458  boundary 9/9
```

Scientific Gate

Outcome:

```text
FAIL
```

Reason:

```text
talos_variants_do_not_satisfy_both_backbone_gate
```

Backbone dispositions:

```text
EEGConformerMini:
  best clean TALOS variant: TALOS_L
  delta bAcc vs ERM: +0.0021
  delta bAcc vs TTA_CONTROL_REPLAY: +0.0021
  gate status: FAIL, below +0.020 threshold

EEGNetMini:
  best post-hoc TALOS variant: TALOS_LD
  delta bAcc vs ERM: +0.0577
  delta bAcc vs TTA_CONTROL_REPLAY: +0.0577
  gate status: FAIL, best gain is boundary-hit diagnostic-only
```

TALOS_L on EEGNetMini narrowly clears the +0.020 bAcc threshold, but it still
has a boundary hit in 1/9 folds. TALOS_D and TALOS_LD show larger EEGNetMini
gains but hit trust-region bounds in 9/9 folds. Under the PM contract, these
cannot count as positive pass evidence.

Collapse / Trust Region

Collapse guards passed for all aggregate variants:

```text
collapse_warning_count: 0 for every backbone x variant
```

Trust-region boundary hits are the limiting issue:

```text
EEGConformerMini TALOS_D:  9/9 boundary-hit
EEGConformerMini TALOS_LD: 9/9 boundary-hit
EEGNetMini TALOS_L:        1/9 boundary-hit
EEGNetMini TALOS_D:        9/9 boundary-hit
EEGNetMini TALOS_LD:       9/9 boundary-hit
```

Interpretation

TALOS_00B is a useful negative diagnostic. The red-team contract held on real
features, and low-dimensional adapters can change target metrics without target
label leakage. However, the strict scientific gate is not satisfied:

```text
no both-backbone +0.020 clean bAcc pass
no boundary-hit positive evidence
no P1 request
no source-free deployment claim
```

Post-hoc readout only; no variant was selected for deployment.

P1 / P2 Status

```text
TALOS_01: BLOCKED
P1 source-free serialized-state training: BLOCKED
P2 streaming / clinical transfer: BLOCKED
source-free deployment claim: BLOCKED
safety / privacy / generalization claim: FORBIDDEN
```
