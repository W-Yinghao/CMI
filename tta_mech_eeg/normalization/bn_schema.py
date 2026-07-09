"""Schema for future normalization / BN audit machinery.

TTA_MECH_02B0 defines contracts only. It does not run model forward passes,
refresh BN, or compute target metrics.
"""

from __future__ import annotations

from typing import Any

from tta_mech_eeg.baselines.registry import stable_hash


REQUIRED_CONDITION_API_PARAMETERS: tuple[str, ...] = (
    "model_or_state",
    "source_state",
    "target_x",
    "condition_config",
)

FORBIDDEN_CONDITION_API_PARAMETERS: tuple[str, ...] = (
    "target_y",
    "y_target",
    "target_metric",
    "target_selected_condition",
)


def bn_audit_schema_payload() -> dict[str, Any]:
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_02B0_normalization_bn_audit_preflight",
        "condition_api": {
            "callable": "run_condition",
            "required_parameters": list(REQUIRED_CONDITION_API_PARAMETERS),
            "forbidden_parameters": list(FORBIDDEN_CONDITION_API_PARAMETERS),
            "target_labels_allowed": False,
            "target_metrics_allowed": False,
        },
        "mutation_rules": {
            "real_forward_run": False,
            "bn_refresh_run": False,
            "target_metrics_computed": False,
            "optimizer_step_allowed": False,
            "loss_backward_allowed": False,
            "encoder_weight_update_allowed": False,
            "classifier_weight_update_allowed": False,
            "checkpoint_overwrite_allowed": False,
            "target_bn_refresh_copy_only": {
                "diagnostic_only": True,
                "copy_required": True,
                "only_bn_buffers_mutable": True,
                "dropout_disabled_required": True,
                "parameters_require_grad_false_required": True,
            },
        },
        "artifact_requirements_for_ready_02b": {
            "has_model_checkpoint": True,
            "has_classifier_head": True,
            "has_bn_buffers": True,
            "has_source_split": True,
            "has_target_X": True,
            "has_source_X": True,
            "has_raw_or_preprocessed_input": True,
            "can_forward_model": True,
            "can_copy_model_without_mutation": True,
            "can_recompute_bn_buffers_on_copy": True,
            "can_disable_dropout": True,
            "can_eval_frozen_bn": True,
            "can_emit_logits_without_target_y": True,
        },
    }
    payload["bn_audit_schema_hash"] = stable_hash(payload)
    return payload
