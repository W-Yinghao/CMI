"""Frozen audit-axis output schema for TTA-MECH."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tta_mech_eeg.baselines.registry import stable_hash


@dataclass(frozen=True)
class AuditAxis:
    name: str
    category: str
    required_fields: tuple[str, ...]
    target_labels_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_axes() -> tuple[AuditAxis, ...]:
    return (
        AuditAxis(
            name="entropy_confidence",
            category="entropy / confidence",
            required_fields=("entropy_before_after", "mean_max_probability", "margin_shift"),
            target_labels_required=False,
        ),
        AuditAxis(
            name="balance_prior",
            category="balance / prior",
            required_fields=("predicted_class_marginal", "source_label_prior", "class_collapse_guard"),
            target_labels_required=False,
        ),
        AuditAxis(
            name="geometry",
            category="feature geometry / recentering",
            required_fields=("feature_mean_shift", "covariance_distance", "recenter_magnitude"),
            target_labels_required=False,
        ),
        AuditAxis(
            name="source_replay",
            category="source replay contribution",
            required_fields=("source_CE_retention", "source_prediction_drift"),
            target_labels_required=False,
        ),
        AuditAxis(
            name="normalization_batchnorm",
            category="normalization / BatchNorm",
            required_fields=("normalization_stat_shift", "BN_or_norm_update_magnitude"),
            target_labels_required=False,
        ),
        AuditAxis(
            name="calibration",
            category="accuracy vs calibration",
            required_fields=("NLL", "ECE", "temperature_equivalent_effect", "bAcc_NLL_divergence"),
            target_labels_required=True,
        ),
    )


def audit_axis_schema_payload() -> dict[str, Any]:
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_00A_artifact_inventory_replay_harness_preflight",
        "axes": [axis.to_dict() for axis in audit_axes()],
        "target_labels_rule": "final metrics and mechanism stratification only after replay",
        "new_method_claim": False,
    }
    payload["audit_axis_schema_hash"] = stable_hash(payload)
    return payload
