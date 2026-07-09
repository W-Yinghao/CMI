"""Read-only artifact inventory for TTA-MECH.

Inventory reports availability and schema hints. It must not trigger real EEG
replay or target metric computation.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tta_mech_eeg.baselines.registry import stable_hash


NPZ_KEY_SUFFIX = ".npy"


@dataclass(frozen=True)
class ArtifactInventoryRecord:
    artifact_type: str
    path: str
    hash: str
    dataset: str
    backbone: str
    fold: str
    seed: str
    contains_z: bool
    contains_y: bool
    contains_domain: bool
    contains_groups: bool
    contains_predictions: bool
    contains_logits: bool
    contains_probs: bool
    usable_for_replay_axis: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _npz_keys(path: str | Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            keys = set()
            for name in zf.namelist():
                if name.endswith(NPZ_KEY_SUFFIX):
                    keys.add(name[: -len(NPZ_KEY_SUFFIX)])
            return keys
    except zipfile.BadZipFile:
        return set()


def _usable_axes(keys: set[str]) -> tuple[str, ...]:
    axes = []
    if {"z", "y"} <= keys:
        axes.append("final_metric_after_replay")
    if "z" in keys:
        axes.append("geometry")
        axes.append("normalization_feature_state")
    if "y" in keys:
        axes.append("source_prior_or_final_label_audit")
    if "logits" in keys or "probs" in keys or "probabilities" in keys:
        axes.append("entropy_confidence_calibration")
    if {"domain", "groups"} <= keys:
        axes.append("split_and_group_audit")
    return tuple(axes)


def inventory_cedar01f_handoff(handoff_manifest: str | Path) -> list[ArtifactInventoryRecord]:
    with Path(handoff_manifest).open() as f:
        handoff = json.load(f)
    records: list[ArtifactInventoryRecord] = []
    for rec in sorted(
        handoff.get("per_artifact_hashes", []),
        key=lambda r: (str(r.get("backbone", "")), int(r.get("fold_id", 0)), str(r.get("path", ""))),
    ):
        path = Path(rec["path"])
        if not path.exists():
            records.append(
                ArtifactInventoryRecord(
                    artifact_type="CEDAR_01F_frozen_feature_npz",
                    path=str(path),
                    hash="",
                    dataset=str(rec.get("dataset", "")),
                    backbone=str(rec.get("backbone", "")),
                    fold=str(rec.get("fold_id", "")),
                    seed=str(rec.get("seed", "")),
                    contains_z=False,
                    contains_y=False,
                    contains_domain=False,
                    contains_groups=False,
                    contains_predictions=False,
                    contains_logits=False,
                    contains_probs=False,
                    usable_for_replay_axis=(),
                    status="MISSING",
                )
            )
            continue
        observed = sha256_file(path)
        keys = _npz_keys(path)
        status = "AVAILABLE" if observed == rec.get("file_sha256") else "REJECTED"
        records.append(
            ArtifactInventoryRecord(
                artifact_type="CEDAR_01F_frozen_feature_npz",
                path=str(path),
                hash=observed,
                dataset=str(rec.get("dataset", "")),
                backbone=str(rec.get("backbone", "")),
                fold=str(rec.get("fold_id", "")),
                seed=str(rec.get("seed", "")),
                contains_z="z" in keys,
                contains_y="y" in keys,
                contains_domain="domain" in keys,
                contains_groups="groups" in keys,
                contains_predictions=bool({"pred", "predictions"} & keys),
                contains_logits="logits" in keys,
                contains_probs=bool({"probs", "probabilities", "p"} & keys),
                usable_for_replay_axis=_usable_axes(keys),
                status=status,
            )
        )
    return records


def detect_existing_summary_candidates(root: str | Path = ".") -> list[ArtifactInventoryRecord]:
    root = Path(root)
    patterns = ("*CITA*", "*TTA*", "*CORAL*", "*SPDIM*", "*T3A*")
    paths: set[Path] = set()
    for base in (root / "results", root / "analysis", root / "reports"):
        if not base.exists():
            continue
        for pattern in patterns:
            paths.update(p for p in base.rglob(pattern) if p.is_file())
    records = []
    for path in sorted(paths):
        records.append(
            ArtifactInventoryRecord(
                artifact_type="existing_summary_candidate",
                path=str(path),
                hash=sha256_file(path),
                dataset="",
                backbone="",
                fold="",
                seed="",
                contains_z=False,
                contains_y=False,
                contains_domain=False,
                contains_groups=False,
                contains_predictions=False,
                contains_logits=False,
                contains_probs=False,
                usable_for_replay_axis=("summary_audit",),
                status="AVAILABLE",
            )
        )
    return records


def build_artifact_inventory(handoff_manifest: str | Path) -> dict[str, Any]:
    records = inventory_cedar01f_handoff(handoff_manifest)
    records.extend(detect_existing_summary_candidates("."))
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_00A_artifact_inventory_replay_harness_preflight",
        "real_eeg_replay_run": False,
        "target_metrics_computed": False,
        "records": [record.to_dict() for record in records],
        "summary": {
            "total_records": len(records),
            "cedar01f_feature_records": sum(r.artifact_type == "CEDAR_01F_frozen_feature_npz" for r in records),
            "available_records": sum(r.status == "AVAILABLE" for r in records),
            "rejected_records": sum(r.status == "REJECTED" for r in records),
            "missing_records": sum(r.status == "MISSING" for r in records),
        },
    }
    payload["artifact_inventory_hash"] = stable_hash(payload)
    return payload
