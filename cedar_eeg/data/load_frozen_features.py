"""Strict loader for compliant CEDAR frozen feature dumps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .feature_schema import (
    DOMAIN_KEYS,
    GROUP_KEYS,
    ROLE_KEYS,
    SUBJECT_KEYS,
    Y_KEYS,
    Z_KEYS,
    find_key,
    sha256_file,
    stable_json_hash,
)


class FrozenFeatureSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenFeatureBundle:
    z: np.ndarray
    y: np.ndarray
    domain: np.ndarray
    groups: np.ndarray
    role: np.ndarray
    metadata: dict[str, Any]

    def diagnostic_full_view(self) -> dict[str, np.ndarray]:
        return {
            "z": self.z,
            "y": self.y,
            "domain": self.domain,
            "groups": self.groups,
            "role": self.role,
        }

    def source_selection_view(self) -> dict[str, np.ndarray]:
        role = np.asarray(self.role).astype(str)
        keep = np.char.startswith(role, "source_")
        if not np.any(keep):
            raise FrozenFeatureSchemaError("source_selection_view has no source rows")
        return {
            "z": self.z[keep],
            "y": self.y[keep],
            "domain": self.domain[keep],
            "groups": self.groups[keep],
        }


def _get(data: np.lib.npyio.NpzFile, aliases: tuple[str, ...], required_name: str) -> np.ndarray:
    key = find_key(set(data.files), aliases)
    if key is None:
        raise FrozenFeatureSchemaError(f"missing {required_name}")
    return np.asarray(data[key])


def _scalar_metadata(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return None


def _validate_lengths(*arrays: np.ndarray) -> None:
    lengths = [int(a.shape[0]) for a in arrays]
    if len(set(lengths)) != 1:
        raise FrozenFeatureSchemaError(f"len mismatch: {lengths}")


def load_frozen_feature_npz(path: str | Path) -> FrozenFeatureBundle:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    z = _get(data, Z_KEYS, "z")
    y = _get(data, Y_KEYS, "y")
    domain = _get(data, DOMAIN_KEYS, "domain")
    groups = _get(data, GROUP_KEYS, "groups")
    role = _get(data, ROLE_KEYS, "role")

    if z.ndim != 2:
        raise FrozenFeatureSchemaError("z must be 2D")
    if not np.issubdtype(z.dtype, np.floating):
        raise FrozenFeatureSchemaError("z must be float32/float64")
    if not np.isfinite(z).all():
        raise FrozenFeatureSchemaError("z has NaN or Inf")
    _validate_lengths(z, y, domain, groups, role)
    if len(np.unique(groups)) < 2:
        raise FrozenFeatureSchemaError("groups must contain at least two groups")
    if len(np.unique(domain)) < 2:
        raise FrozenFeatureSchemaError("domain must contain at least two values")

    sample_id_key = find_key(set(data.files), ("sample_id",))
    if sample_id_key is not None:
        sample_id = np.asarray(data[sample_id_key])
        _validate_lengths(z, sample_id)
        if np.array_equal(sample_id.astype(str), groups.astype(str)):
            justification = _scalar_metadata(data, "grouping_justification")
            if not justification:
                raise FrozenFeatureSchemaError(
                    "groups equal sample_id for every row without grouping_justification"
                )

    role_text = role.astype(str)
    if not np.any(np.char.startswith(role_text, "source_")):
        raise FrozenFeatureSchemaError("role must include at least one source_* row")

    metadata = {
        "path": str(path),
        "dataset": _scalar_metadata(data, "dataset"),
        "backbone": _scalar_metadata(data, "backbone"),
        "seed": _scalar_metadata(data, "seed"),
        "fold_id": _scalar_metadata(data, "fold_id"),
        "file_sha256": sha256_file(path),
        "schema_version": "CEDAR_01F_v1",
    }
    metadata["manifest_hash"] = stable_json_hash(metadata)
    return FrozenFeatureBundle(
        z=z.astype(np.float64, copy=False),
        y=y.astype(np.int64, copy=False),
        domain=domain,
        groups=groups,
        role=role_text,
        metadata=metadata,
    )


def write_feature_manifest(bundle: FrozenFeatureBundle, out_path: str | Path) -> dict[str, Any]:
    manifest = {
        **bundle.metadata,
        "n_samples": int(bundle.z.shape[0]),
        "z_dim": int(bundle.z.shape[1]),
        "deployable": False,
        "cedar_role": "feature_supply_candidate_only",
    }
    manifest["manifest_hash"] = stable_json_hash(manifest)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def verify_manifest_immutability(path: str | Path, expected_file_sha256: str) -> None:
    observed = sha256_file(path)
    if observed != expected_file_sha256:
        raise FrozenFeatureSchemaError("feature dump sha256 does not match frozen manifest")
