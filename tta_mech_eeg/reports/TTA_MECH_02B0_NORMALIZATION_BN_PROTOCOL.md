TTA_MECH_02B0 - Normalization / BN Audit Preflight Protocol

This phase is a preflight only. It does not run real EEG forward passes, does
not refresh BN, does not compute target metrics, and does not add a method or
baseline.

Purpose

TTA_MECH_02B0 asks whether the current repository artifacts are sufficient to
run a strict normalization / BN mechanism audit later. It does not answer the
scientific BN question.

Source baseline

```text
TTA_MECH_01S mechanism synthesis commit: 7e0ddc4
source feature handoff: CEDAR_01F BNCI2014_001 seed0
dataset: BNCI2014_001
backbones: EEGNetMini, EEGConformerMini
folds: 9 per backbone
```

Frozen condition universe

```text
ERM_FROZEN_EVAL
SOURCE_BN_REPLAY_IF_AVAILABLE
TARGET_BN_REFRESH_COPY_ONLY
FEATURE_SOURCE_NORMALIZATION
FEATURE_TARGET_RECENTER_DIAGNOSTIC
MATCHED_CORAL_EXISTING_BASELINE_REFERENCE
SPDIM_EXISTING_BASELINE_REFERENCE
```

`TARGET_BN_REFRESH_COPY_ONLY` is diagnostic only. It may mutate BN buffers only
on a copied model, never the original checkpoint and never model weights. It
must use no target labels and must not select a deployment condition.

Future condition API

```text
run_condition(model_or_state, source_state, target_x, condition_config)
```

Forbidden API parameters:

```text
target_y
y_target
target_metric
target_selected_condition
```

Ready criteria for a future 02B audit

For at least one full LOSO backbone, every fold must have:

```text
has_model_checkpoint
has_classifier_head
has_bn_buffers
has_source_split
has_target_X
has_source_X
has_raw_or_preprocessed_input
can_forward_model
can_copy_model_without_mutation
can_recompute_bn_buffers_on_copy
can_disable_dropout
can_eval_frozen_bn
can_emit_logits_without_target_y
```

If these are unavailable, the correct preflight result is:

```text
TTA_MECH_02B_NOT_FEASIBLE_FROM_CURRENT_ARTIFACTS
```

Red-team requirements

```text
target-label quarantine
condition universe freeze
no weight update
dropout / train-mode guard
artifact immutability
no new method guard
```

Forbidden scope

```text
TTA_MECH_02B real run
new adaptation baseline
new adapter
TALOS / CEDAR / CMI / CutClean rescue
pruning / mask / surgery
source-free deployment claim
deployment baseline selection
P1/P2 training
```
