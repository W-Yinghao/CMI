"""Frozen condition registry for TTA_MECH_02B0 normalization / BN preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tta_mech_eeg.baselines.registry import stable_hash


ALLOWED_CONDITIONS: tuple[str, ...] = (
    "ERM_FROZEN_EVAL",
    "SOURCE_BN_REPLAY_IF_AVAILABLE",
    "TARGET_BN_REFRESH_COPY_ONLY",
    "FEATURE_SOURCE_NORMALIZATION",
    "FEATURE_TARGET_RECENTER_DIAGNOSTIC",
    "MATCHED_CORAL_EXISTING_BASELINE_REFERENCE",
    "SPDIM_EXISTING_BASELINE_REFERENCE",
)

FORBIDDEN_CONDITIONS: tuple[str, ...] = (
    "ADAPTIVE_TRAINING",
    "LEARNED_ADAPTER",
    "NEW_TTA_METHOD",
    "SAFETY_GATE",
    "HARM_ROUTER",
    "CMI_REGULARIZER",
    "PRUNING",
    "CUT_CLEAN",
    "TALOS",
    "CEDAR",
)


@dataclass(frozen=True)
class ConditionEntry:
    name: str
    type: str
    diagnostic_only: bool
    requires_model_checkpoint: bool
    requires_classifier_head: bool
    requires_bn_buffers: bool
    requires_source_split: bool
    requires_target_x: bool
    requires_source_x: bool
    requires_raw_or_preprocessed_input: bool
    can_emit_logits_without_target_y: bool
    target_labels_allowed: bool
    deployment_selection_allowed: bool
    mutates_original_model_allowed: bool
    mutates_weights_allowed: bool
    mutates_bn_buffers_allowed: bool
    mutates_feature_normalizer_allowed: bool
    copy_only_mutation_required: bool
    train_mode_allowed: bool
    dropout_disabled_required: bool
    parameters_require_grad_false_required: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def condition_registry() -> tuple[ConditionEntry, ...]:
    return (
        ConditionEntry(
            name="ERM_FROZEN_EVAL",
            type="frozen_model_diagnostic",
            diagnostic_only=True,
            requires_model_checkpoint=True,
            requires_classifier_head=True,
            requires_bn_buffers=True,
            requires_source_split=False,
            requires_target_x=True,
            requires_source_x=False,
            requires_raw_or_preprocessed_input=True,
            can_emit_logits_without_target_y=True,
            target_labels_allowed=False,
            deployment_selection_allowed=False,
            mutates_original_model_allowed=False,
            mutates_weights_allowed=False,
            mutates_bn_buffers_allowed=False,
            mutates_feature_normalizer_allowed=False,
            copy_only_mutation_required=False,
            train_mode_allowed=False,
            dropout_disabled_required=True,
            parameters_require_grad_false_required=True,
            notes="Frozen checkpoint eval if checkpoint, BN buffers, classifier, and target X exist.",
        ),
        ConditionEntry(
            name="SOURCE_BN_REPLAY_IF_AVAILABLE",
            type="source_state_diagnostic",
            diagnostic_only=True,
            requires_model_checkpoint=True,
            requires_classifier_head=True,
            requires_bn_buffers=True,
            requires_source_split=True,
            requires_target_x=True,
            requires_source_x=True,
            requires_raw_or_preprocessed_input=True,
            can_emit_logits_without_target_y=True,
            target_labels_allowed=False,
            deployment_selection_allowed=False,
            mutates_original_model_allowed=False,
            mutates_weights_allowed=False,
            mutates_bn_buffers_allowed=False,
            mutates_feature_normalizer_allowed=False,
            copy_only_mutation_required=True,
            train_mode_allowed=False,
            dropout_disabled_required=True,
            parameters_require_grad_false_required=True,
            notes="Replay existing source BN state on a copy when full source/raw artifacts exist.",
        ),
        ConditionEntry(
            name="TARGET_BN_REFRESH_COPY_ONLY",
            type="target_unlabeled_bn_diagnostic",
            diagnostic_only=True,
            requires_model_checkpoint=True,
            requires_classifier_head=True,
            requires_bn_buffers=True,
            requires_source_split=False,
            requires_target_x=True,
            requires_source_x=False,
            requires_raw_or_preprocessed_input=True,
            can_emit_logits_without_target_y=True,
            target_labels_allowed=False,
            deployment_selection_allowed=False,
            mutates_original_model_allowed=False,
            mutates_weights_allowed=False,
            mutates_bn_buffers_allowed=True,
            mutates_feature_normalizer_allowed=False,
            copy_only_mutation_required=True,
            train_mode_allowed=True,
            dropout_disabled_required=True,
            parameters_require_grad_false_required=True,
            notes="Diagnostic only: BN buffers may change on a copied model, never original weights.",
        ),
        ConditionEntry(
            name="FEATURE_SOURCE_NORMALIZATION",
            type="feature_stat_diagnostic",
            diagnostic_only=True,
            requires_model_checkpoint=False,
            requires_classifier_head=False,
            requires_bn_buffers=False,
            requires_source_split=True,
            requires_target_x=False,
            requires_source_x=False,
            requires_raw_or_preprocessed_input=False,
            can_emit_logits_without_target_y=False,
            target_labels_allowed=False,
            deployment_selection_allowed=False,
            mutates_original_model_allowed=False,
            mutates_weights_allowed=False,
            mutates_bn_buffers_allowed=False,
            mutates_feature_normalizer_allowed=True,
            copy_only_mutation_required=True,
            train_mode_allowed=False,
            dropout_disabled_required=True,
            parameters_require_grad_false_required=True,
            notes="Feature-stat diagnostic only; no forward path and no target labels.",
        ),
        ConditionEntry(
            name="FEATURE_TARGET_RECENTER_DIAGNOSTIC",
            type="feature_stat_diagnostic",
            diagnostic_only=True,
            requires_model_checkpoint=False,
            requires_classifier_head=False,
            requires_bn_buffers=False,
            requires_source_split=False,
            requires_target_x=False,
            requires_source_x=False,
            requires_raw_or_preprocessed_input=False,
            can_emit_logits_without_target_y=False,
            target_labels_allowed=False,
            deployment_selection_allowed=False,
            mutates_original_model_allowed=False,
            mutates_weights_allowed=False,
            mutates_bn_buffers_allowed=False,
            mutates_feature_normalizer_allowed=True,
            copy_only_mutation_required=True,
            train_mode_allowed=False,
            dropout_disabled_required=True,
            parameters_require_grad_false_required=True,
            notes="Frozen-feature recentering diagnostic; not a BN forward audit.",
        ),
        ConditionEntry(
            name="MATCHED_CORAL_EXISTING_BASELINE_REFERENCE",
            type="existing_baseline_reference",
            diagnostic_only=True,
            requires_model_checkpoint=False,
            requires_classifier_head=False,
            requires_bn_buffers=False,
            requires_source_split=True,
            requires_target_x=False,
            requires_source_x=False,
            requires_raw_or_preprocessed_input=False,
            can_emit_logits_without_target_y=False,
            target_labels_allowed=False,
            deployment_selection_allowed=False,
            mutates_original_model_allowed=False,
            mutates_weights_allowed=False,
            mutates_bn_buffers_allowed=False,
            mutates_feature_normalizer_allowed=False,
            copy_only_mutation_required=False,
            train_mode_allowed=False,
            dropout_disabled_required=True,
            parameters_require_grad_false_required=True,
            notes="Existing matched-CORAL replay reference from TTA_MECH_01, not a new run.",
        ),
        ConditionEntry(
            name="SPDIM_EXISTING_BASELINE_REFERENCE",
            type="existing_baseline_reference",
            diagnostic_only=True,
            requires_model_checkpoint=False,
            requires_classifier_head=False,
            requires_bn_buffers=False,
            requires_source_split=True,
            requires_target_x=False,
            requires_source_x=False,
            requires_raw_or_preprocessed_input=False,
            can_emit_logits_without_target_y=False,
            target_labels_allowed=False,
            deployment_selection_allowed=False,
            mutates_original_model_allowed=False,
            mutates_weights_allowed=False,
            mutates_bn_buffers_allowed=False,
            mutates_feature_normalizer_allowed=False,
            copy_only_mutation_required=False,
            train_mode_allowed=False,
            dropout_disabled_required=True,
            parameters_require_grad_false_required=True,
            notes="Existing SPDIM replay reference from TTA_MECH_01, not a new run.",
        ),
    )


def condition_registry_payload() -> dict[str, Any]:
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_02B0_normalization_bn_audit_preflight",
        "allowed_conditions": list(ALLOWED_CONDITIONS),
        "forbidden_conditions": list(FORBIDDEN_CONDITIONS),
        "entries": [entry.to_dict() for entry in condition_registry()],
        "runtime_addition_allowed": False,
        "target_labels_allowed": False,
        "deployment_selection_allowed": False,
        "real_forward_run": False,
        "bn_refresh_run": False,
        "target_metrics_computed": False,
    }
    payload["condition_registry_hash"] = stable_hash(payload)
    return payload
