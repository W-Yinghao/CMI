"""ACAR V5 Stage-1B FILE-backed artifact writer (stdlib). For a REAL build the trainer emits large output FILES, not in-memory
bytes; this writer streams sha256 over each file (rejecting missing/empty), IGNORES any trainer-reported hash, and emits the same
validated 6-field registry manifest. Synthetic-tested with temp files (no DEV, no model).
"""
from __future__ import annotations
import hashlib
import os
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_artifacts as ART

# each registry hash field is computed by streaming the file at this path key in the trainer's raw output
FILE_SOURCE = {
    "encoder_state_dict_sha256": "encoder_state_dict_path",
    "encoder_checkpoint_file_sha256": "encoder_checkpoint_file_path",
    "source_state_artifact_sha256": "source_state_artifact_path",
    "source_state_file_sha256": "source_state_file_path",
    "preprocessing_config_sha256": "preprocessing_config_path",
    "feat_dump_sha256": "feat_dump_path",
}
assert set(FILE_SOURCE) == set(P.REGISTRY_HASH_FIELDS)


class Stage1bFileArtifactError(RuntimeError):
    """Raised when a trainer's file outputs are missing / empty / mis-specified."""


def _sha256_stream(path):
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            total += len(chunk)
    if total == 0:
        raise Stage1bFileArtifactError(f"empty artifact file: {path}")
    return h.hexdigest()


def write_artifact_from_files(raw, *, expected_ref, disease, fold, seed):
    """Compute the artifact manifest by streaming sha256 over the trainer's output files. Any hash string in `raw` is IGNORED."""
    if not isinstance(raw, dict):
        raise Stage1bFileArtifactError("raw build output must be a dict")
    if raw.get("ref") != expected_ref:
        raise Stage1bFileArtifactError(f"raw ref {raw.get('ref')!r} != expected {expected_ref!r}")
    art = {"ref": expected_ref, "disease": disease, "fold": fold, "seed": seed}
    for hash_field, path_key in FILE_SOURCE.items():
        path = raw.get(path_key)
        if not isinstance(path, str) or not path:
            raise Stage1bFileArtifactError(f"{expected_ref}: raw['{path_key}'] must be a non-empty path")
        if not os.path.isfile(path):
            raise Stage1bFileArtifactError(f"{expected_ref}: artifact file missing: {path}")
        art[hash_field] = _sha256_stream(path)                # streamed from the actual file bytes (computed, not trusted)
    ART.validate_artifact_manifest(art, expected_ref=expected_ref, disease=disease, fold=fold, seed=seed)
    return art
