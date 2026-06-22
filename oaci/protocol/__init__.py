"""OACI confirmatory protocol: manifest + freeze + runner gate (no paper-level defaults in code)."""
from __future__ import annotations

from .manifest import REQUIRED, ProtocolManifest
from .freeze import default_confirmatory_path, freeze, load_yaml_manifest
from .confirmatory import RunEvidence, assert_confirmatory_runnable, collect_evidence, confirmatory_refusals
from .manifest_v2 import DatasetBlock, ProtocolManifestV2, load_v2

__all__ = [
    "REQUIRED", "ProtocolManifest",
    "default_confirmatory_path", "freeze", "load_yaml_manifest",
    "RunEvidence", "assert_confirmatory_runnable", "collect_evidence", "confirmatory_refusals",
    "DatasetBlock", "ProtocolManifestV2", "load_v2",
]
