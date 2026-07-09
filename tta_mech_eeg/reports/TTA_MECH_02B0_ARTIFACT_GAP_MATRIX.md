TTA_MECH_02B0N - Artifact Gap Matrix

This matrix summarizes observed artifact support from the accepted 02B0
preflight inventory. It is not a new inventory run and it does not start 02B.

Source files

```text
results/tta_mech/tta_mech02b0_preflight/artifact_inventory.json
results/tta_mech/tta_mech02b0_preflight/artifact_inventory.csv
```

Aggregate gap matrix

| Field | Observed support | 02B implication |
| --- | --- | --- |
| has_model_checkpoint | false, 0/18 | Cannot load model state for BN audit. |
| has_classifier_head | false, 0/18 | Cannot emit checkpoint-level logits. |
| has_bn_buffers | false, 0/18 | Cannot compare frozen/source/target BN states. |
| has_feature_normalizer | false, 0/18 | No retained feature normalizer artifact. |
| has_source_X | false, 0/18 | Cannot replay source forward or recompute source statistics. |
| has_target_X | false, 0/18 | Cannot run target forward or target BN diagnostic. |
| has_raw_or_preprocessed_input | false, 0/18 | No input tensors for deterministic forward path. |
| can_forward_model | false, 0/18 | Real 02B audit cannot run. |
| can_copy_model_without_mutation | false, 0/18 | Copy-only mutation isolation cannot be tested. |
| can_recompute_bn_buffers_on_copy | false, 0/18 | Target/source BN refresh diagnostic cannot run. |
| can_disable_dropout | false, 0/18 | Dropout-disabled train-mode BN refresh cannot be guaranteed. |
| can_eval_frozen_bn | false, 0/18 | Frozen-BN forward comparison cannot run. |
| can_emit_logits_without_target_y | false, 0/18 | Label-free forward logits cannot be produced. |

Backbone readiness

| Backbone | Folds total | Folds ready | Full LOSO ready | Status |
| --- | ---: | ---: | --- | --- |
| EEGConformerMini | 9 | 0 | false | NOT_READY |
| EEGNetMini | 9 | 0 | false | NOT_READY |

Readiness interpretation

Current artifacts are frozen feature outputs. They include enough information
for feature-space replay, geometry/recentering summaries, calibration metrics,
entropy/confidence summaries, and class-balance/prior audit. They do not include
the model-level state needed for a BN / normalization causal audit.

Hard boundary

Filling these gaps would require new artifact supply or reconstruction:
checkpoints, classifier heads, BN buffers, raw/preprocessed source and target
inputs, and a deterministic forward path. That work is not approved under
TTA_MECH_02B0N.
