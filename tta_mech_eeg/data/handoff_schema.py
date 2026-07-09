"""Read-only handoff availability schema for TTA-MECH."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tta_mech_eeg.baselines.registry import stable_hash
from tta_mech_eeg.data.artifact_inventory import sha256_file


DEFAULT_CEDAR01F_HANDOFF = Path(
    "results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/CEDAR_01F_HANDOFF_MANIFEST.json"
)


@dataclass(frozen=True)
class HandoffAvailability:
    path: str
    exists: bool
    file_hash: str
    expected_artifacts: int
    listed_artifacts: int
    status: str
    real_replay_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def validate_cedar01f_handoff_availability(path: str | Path = DEFAULT_CEDAR01F_HANDOFF) -> HandoffAvailability:
    path = Path(path)
    if not path.exists():
        return HandoffAvailability(
            path=str(path),
            exists=False,
            file_hash="",
            expected_artifacts=18,
            listed_artifacts=0,
            status="MISSING",
        )
    payload = load_json(path)
    artifacts = payload.get("per_artifact_hashes", [])
    status = "AVAILABLE" if payload.get("handoff_manifest_is_canonical") is True and len(artifacts) == 18 else "REJECTED"
    return HandoffAvailability(
        path=str(path),
        exists=True,
        file_hash=sha256_file(path),
        expected_artifacts=18,
        listed_artifacts=len(artifacts),
        status=status,
    )


def handoff_schema_payload(path: str | Path = DEFAULT_CEDAR01F_HANDOFF) -> dict[str, Any]:
    availability = validate_cedar01f_handoff_availability(path)
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_00A_artifact_inventory_replay_harness_preflight",
        "cedar01f_handoff": availability.to_dict(),
        "real_replay_run": False,
    }
    payload["handoff_schema_hash"] = stable_hash(payload)
    return payload
