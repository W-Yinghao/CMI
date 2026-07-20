"""Freeze a protocol manifest to canonical JSON + SHA-256, and load one from YAML."""
from __future__ import annotations

import hashlib
import os

from .manifest import REQUIRED, ProtocolManifest


def freeze(manifest: ProtocolManifest) -> dict:
    manifest.validate_complete()
    canon = manifest.to_canonical_json()
    return {"canonical_json": canon, "sha256": hashlib.sha256(canon.encode()).hexdigest()}


def load_yaml_manifest(path: str) -> ProtocolManifest:
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PyYAML required to load {path}: {e}")
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    return ProtocolManifest(**{k: d.get(k) for k in REQUIRED})


def default_confirmatory_path() -> str:
    return os.path.join(os.path.dirname(__file__), "confirmatory_v1.yaml")
