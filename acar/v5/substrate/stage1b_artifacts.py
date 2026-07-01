"""ACAR V5 Stage-1B artifact-manifest validation (pure/stdlib; no I/O). Each built fold substrate must emit an artifact manifest
carrying the COMPLETE V5 registry hash set (so it is admissible into the substrate registry) + its (ref, disease, fold, seed).
"""
from __future__ import annotations
from acar.v5 import protocol as P

_HEX = "0123456789abcdef"
ARTIFACT_REQUIRED_HASHES = P.REGISTRY_HASH_FIELDS   # encoder_state_dict/checkpoint_file/source_state_artifact/source_state_file/preprocessing_config/feat_dump


class Stage1bArtifactError(RuntimeError):
    """Raised when a built artifact manifest is incomplete / mis-keyed / non-hex."""


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(c in _HEX for c in s.lower())


def validate_artifact_manifest(art, *, expected_ref, disease, fold, seed):
    """Fail-closed: the artifact manifest must match its (ref, disease, fold, seed) and carry the COMPLETE registry hash set."""
    if not isinstance(art, dict):
        raise Stage1bArtifactError("artifact manifest must be a dict")
    if art.get("ref") != expected_ref:
        raise Stage1bArtifactError(f"artifact ref {art.get('ref')!r} != expected {expected_ref!r}")
    if art.get("disease") != disease or int(art.get("fold", -1)) != int(fold) or int(art.get("seed", -1)) != int(seed):
        raise Stage1bArtifactError(f"{expected_ref}: artifact (disease,fold,seed) mismatch")
    missing = [h for h in ARTIFACT_REQUIRED_HASHES if h not in art]
    if missing:
        raise Stage1bArtifactError(f"{expected_ref}: artifact missing registry hash fields {missing}")
    bad = [h for h in ARTIFACT_REQUIRED_HASHES if not _is_hex64(art[h])]
    if bad:
        raise Stage1bArtifactError(f"{expected_ref}: artifact hash fields not 64-hex {bad}")
    return True
