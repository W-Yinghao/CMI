TTA_MECH_02B0 - Artifact Inventory Readout

This is an artifact inventory only. No real forward pass, BN refresh, or target
metric computation was run.

Inventory root

```text
results/tta_mech/tta_mech02b0_preflight/
```

Inventory result

```text
total_records: 18
ready_records: 0
partial_records: 0
rejected_records: 18
feature_artifact_hashes_match_handoff: true
has_any_model_checkpoint: false
has_any_bn_buffers: false
has_any_raw_or_preprocessed_input: false
has_any_forward_ready_artifact: false
feasibility: TTA_MECH_02B_NOT_FEASIBLE_FROM_CURRENT_ARTIFACTS
```

Backbone readiness

| Backbone | Folds total | Folds ready | Full LOSO ready | Status |
| --- | ---: | ---: | --- | --- |
| EEGConformerMini | 9 | 0 | false | NOT_READY |
| EEGNetMini | 9 | 0 | false | NOT_READY |

Per-record outcome

Every CEDAR_01F frozen-feature artifact has a matching handoff hash and a
source split, but every record is rejected for BN audit readiness because the
current artifacts do not include model checkpoints, classifier heads, BN
buffers, target/source raw or preprocessed X, or a forward-ready path.

Representative reject reason

```text
missing has_model_checkpoint, has_classifier_head, has_bn_buffers, has_target_X,
has_source_X, has_raw_or_preprocessed_input, can_forward_model,
can_copy_model_without_mutation, can_recompute_bn_buffers_on_copy,
can_disable_dropout, can_eval_frozen_bn, can_emit_logits_without_target_y
```

Machine-readable files

```text
artifact_inventory.json
artifact_inventory.csv
bn_artifact_inventory_hash.txt
```

Boundary statement

This is not a negative BN mechanism result. It says only that the current
frozen-feature artifacts are insufficient to run a strict BN / normalization
audit without adding new artifacts or changing protocol.
