"""Frozen existing-baseline registry for TTA-MECH.

The registry is intentionally descriptive. It does not implement methods and
must not be mutated based on data or target metrics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


ALLOWED_BASELINES: tuple[str, ...] = (
    "ERM_NO_ADAPT",
    "TTA_CONTROL_REPLAY",
    "MATCHED_CORAL",
    "SPDIM",
    "T3A",
)

@dataclass(frozen=True)
class BaselineEntry:
    name: str
    type: str
    requires_target_x: bool
    requires_target_y: bool
    requires_source_examples: bool
    requires_source_state: bool
    allowed_in_tta_mech: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def baseline_registry() -> tuple[BaselineEntry, ...]:
    return (
        BaselineEntry(
            name="ERM_NO_ADAPT",
            type="existing_baseline",
            requires_target_x=True,
            requires_target_y=False,
            requires_source_examples=False,
            requires_source_state=True,
            allowed_in_tta_mech=True,
            notes="Frozen source readout evaluated without target-time update.",
        ),
        BaselineEntry(
            name="TTA_CONTROL_REPLAY",
            type="existing_baseline",
            requires_target_x=True,
            requires_target_y=False,
            requires_source_examples=False,
            requires_source_state=True,
            allowed_in_tta_mech=True,
            notes="Historical target-unlabeled control replay family.",
        ),
        BaselineEntry(
            name="MATCHED_CORAL",
            type="existing_baseline",
            requires_target_x=True,
            requires_target_y=False,
            requires_source_examples=True,
            requires_source_state=True,
            allowed_in_tta_mech=True,
            notes="Existing covariance-alignment replay baseline.",
        ),
        BaselineEntry(
            name="SPDIM",
            type="existing_baseline",
            requires_target_x=True,
            requires_target_y=False,
            requires_source_examples=True,
            requires_source_state=True,
            allowed_in_tta_mech=True,
            notes="Existing feature-space recentering replay baseline.",
        ),
        BaselineEntry(
            name="T3A",
            type="existing_baseline",
            requires_target_x=True,
            requires_target_y=False,
            requires_source_examples=False,
            requires_source_state=True,
            allowed_in_tta_mech=True,
            notes="Existing classifier-template adjustment baseline.",
        ),
    )


def registry_payload() -> dict[str, Any]:
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_00A_artifact_inventory_replay_harness_preflight",
        "allowed_baselines": list(ALLOWED_BASELINES),
        "entries": [entry.to_dict() for entry in baseline_registry()],
        "runtime_addition_allowed": False,
        "target_labels_allowed_for": ["final_metrics_after_replay_only"],
    }
    payload["baseline_registry_hash"] = stable_hash(payload)
    return payload


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()
