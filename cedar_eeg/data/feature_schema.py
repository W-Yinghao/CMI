"""CEDAR frozen-feature schema inspection.

This module is intentionally read-only. It inventories feature supply candidates
and records provenance; it does not run CEDAR selection.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


Z_KEYS = ("z", "Z", "features")
Y_KEYS = ("y", "labels")
DOMAIN_KEYS = ("domain", "d", "domains")
GROUP_KEYS = ("groups", "recording", "session")
ROLE_KEYS = ("role",)
SUBJECT_KEYS = ("subject_id", "subject")
SESSION_RECORDING_KEYS = ("session_id", "session", "recording_id", "recording")


@dataclass(frozen=True)
class FeatureInventoryRecord:
    dataset: str
    backbone: str
    fold: str
    seed: str
    path: str
    has_z: bool
    has_y: bool
    has_domain: bool
    has_groups: bool
    has_role: bool
    has_subject: bool
    has_session_or_recording: bool
    n_samples: int | None
    z_dim: int | None
    status: str
    reject_reason: str
    provenance: str
    cedar_role: str
    deployable: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def find_key(keys: set[str], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        if key in keys:
            return key
    return None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def infer_dataset_backbone_seed(path: str | Path, data: np.lib.npyio.NpzFile | None = None) -> tuple[str, str, str]:
    path_text = str(path)
    name = Path(path).name
    dataset = ""
    backbone = ""
    seed = ""
    if data is not None:
        for key in ("dataset", "backbone", "seed"):
            if key in data:
                val = np.asarray(data[key])
                if val.shape == ():
                    text = str(val.item())
                else:
                    text = str(val.reshape(-1)[0])
                if key == "dataset":
                    dataset = text
                elif key == "backbone":
                    backbone = text
                elif key == "seed":
                    seed = text
    if not dataset:
        match = re.search(r"(BNCI20\d{2}_\d{3}|BNCI2014_001|BNCI2015_001|[A-Z]+_[A-Za-z0-9]+|PD_ds\d+|SCZ_ds\d+)", path_text)
        dataset = match.group(1) if match else ""
    if not backbone:
        for token in ("EEGConformer", "Conformer", "EEGNetMini", "EEGNet", "GraphCMI", "TSMNet", "LogCov"):
            if token.lower() in path_text.lower():
                backbone = token
                break
    if not seed:
        match = re.search(r"(?:seed|_s)(\d+)", name)
        seed = match.group(1) if match else ""
    return dataset, backbone, seed


def _array_len(data: np.lib.npyio.NpzFile, key: str | None) -> int | None:
    if key is None:
        return None
    arr = np.asarray(data[key])
    if arr.shape == ():
        return None
    return int(arr.shape[0])


def _z_shape(data: np.lib.npyio.NpzFile, key: str | None) -> tuple[int | None, int | None]:
    if key is None:
        return None, None
    arr = np.asarray(data[key])
    if arr.ndim != 2:
        return _array_len(data, key), None
    return int(arr.shape[0]), int(arr.shape[1])


def _split_key(data_keys: set[str], prefix: str) -> str | None:
    for suffix in ("se", "source", "source_audit", "train"):
        key = f"{prefix}_{suffix}"
        if key in data_keys:
            return key
    return None


def inspect_feature_file(path: str | Path, *, include_archive: bool = False) -> FeatureInventoryRecord:
    path = Path(path)
    provenance = "active_workspace_candidate"
    if "archive" in path.parts:
        provenance = "legacy_archive_diagnostic_candidate"
    cedar_role = "feature_supply_candidate_only" if provenance.startswith("legacy") else "cedar01_feature_supply_candidate"

    if path.suffix != ".npz":
        dataset, backbone, seed = infer_dataset_backbone_seed(path)
        return FeatureInventoryRecord(
            dataset=dataset,
            backbone=backbone,
            fold="",
            seed=seed,
            path=str(path),
            has_z=False,
            has_y=False,
            has_domain=False,
            has_groups=False,
            has_role=False,
            has_subject=False,
            has_session_or_recording=False,
            n_samples=None,
            z_dim=None,
            status="REJECT",
            reject_reason="not_npz_feature_dump",
            provenance=provenance,
            cedar_role=cedar_role,
        )

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - defensive inventory path
        dataset, backbone, seed = infer_dataset_backbone_seed(path)
        return FeatureInventoryRecord(
            dataset=dataset,
            backbone=backbone,
            fold="",
            seed=seed,
            path=str(path),
            has_z=False,
            has_y=False,
            has_domain=False,
            has_groups=False,
            has_role=False,
            has_subject=False,
            has_session_or_recording=False,
            n_samples=None,
            z_dim=None,
            status="REJECT",
            reject_reason=f"npz_load_error:{type(exc).__name__}",
            provenance=provenance,
            cedar_role=cedar_role,
        )

    keys = set(data.files)
    dataset, backbone, seed = infer_dataset_backbone_seed(path, data)
    z_key = find_key(keys, Z_KEYS)
    y_key = find_key(keys, Y_KEYS)
    domain_key = find_key(keys, DOMAIN_KEYS)
    group_key = find_key(keys, GROUP_KEYS)
    role_key = find_key(keys, ROLE_KEYS)
    subject_key = find_key(keys, SUBJECT_KEYS)
    session_key = find_key(keys, SESSION_RECORDING_KEYS)

    legacy_z_key = _split_key(keys, "z")
    legacy_y_key = _split_key(keys, "y")
    legacy_group_key = _split_key(keys, "group_id") or _split_key(keys, "recording_id")
    legacy_subject_key = _split_key(keys, "subject_id")
    legacy_session_key = _split_key(keys, "recording_id")

    has_z = z_key is not None or legacy_z_key is not None
    has_y = y_key is not None or legacy_y_key is not None
    has_domain = domain_key is not None or legacy_subject_key is not None
    has_groups = group_key is not None or legacy_group_key is not None
    has_role = role_key is not None
    has_subject = subject_key is not None or legacy_subject_key is not None
    has_session_or_recording = session_key is not None or legacy_session_key is not None

    n_samples, z_dim = _z_shape(data, z_key or legacy_z_key)
    lengths = [
        _array_len(data, key)
        for key in (z_key, y_key, domain_key, group_key, role_key)
        if key is not None
    ]
    direct_len_ok = bool(lengths) and all(x == lengths[0] for x in lengths if x is not None)

    status = "REJECT"
    reject_reason = ""
    if z_key and y_key and domain_key and group_key and role_key:
        if not direct_len_ok:
            reject_reason = "direct_schema_len_mismatch"
        elif z_dim is None:
            reject_reason = "z_not_2d"
        elif not np.isfinite(np.asarray(data[z_key])).all():
            reject_reason = "z_has_nan_or_inf"
        elif len(np.unique(np.asarray(data[group_key]))) < 2:
            reject_reason = "single_group"
        elif len(np.unique(np.asarray(data[domain_key]))) < 2:
            reject_reason = "single_domain"
        else:
            status = "COMPLIANT"
    elif legacy_z_key and legacy_y_key and legacy_group_key and legacy_subject_key:
        status = "ADAPTER_POSSIBLE"
        reject_reason = "legacy_split_schema_requires_adapter_manifest"
    else:
        missing = []
        if not has_z:
            missing.append("z")
        if not has_y:
            missing.append("y")
        if not has_domain:
            missing.append("domain")
        if not has_groups:
            missing.append("groups")
        if not missing and not has_role:
            missing.append("role")
        reject_reason = "missing_" + "_".join(missing) if missing else "unsupported_schema"

    if not include_archive and provenance.startswith("legacy"):
        cedar_role = "legacy_archive_inventory_only"

    return FeatureInventoryRecord(
        dataset=dataset,
        backbone=backbone,
        fold="",
        seed=seed,
        path=str(path),
        has_z=has_z,
        has_y=has_y,
        has_domain=has_domain,
        has_groups=has_groups,
        has_role=has_role,
        has_subject=has_subject,
        has_session_or_recording=has_session_or_recording,
        n_samples=n_samples,
        z_dim=z_dim,
        status=status,
        reject_reason=reject_reason,
        provenance=provenance,
        cedar_role=cedar_role,
    )
