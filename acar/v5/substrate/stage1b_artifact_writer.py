"""ACAR V5 Stage-1B artifact writer (pure/stdlib). Computes the registry hash set by hashing the trainer's actual output BYTES —
it does NOT trust any hash string the trainer might report. Emits a validated artifact manifest (complete 6-field registry hash
set + matching ref/disease/fold/seed).
"""
from __future__ import annotations
import hashlib
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_artifacts as ART

# each registry hash field is computed over a specific bytes payload in the trainer's raw build output
HASH_SOURCE = {
    "encoder_state_dict_sha256": "encoder_state_dict_bytes",
    "encoder_checkpoint_file_sha256": "encoder_checkpoint_bytes",
    "source_state_artifact_sha256": "source_state_artifact_bytes",
    "source_state_file_sha256": "source_state_file_bytes",
    "preprocessing_config_sha256": "preprocessing_config_bytes",
    "feat_dump_sha256": "feat_dump_bytes",
}
assert set(HASH_SOURCE) == set(P.REGISTRY_HASH_FIELDS)


class Stage1bArtifactWriteError(RuntimeError):
    """Raised when a trainer's raw build output is incomplete / non-bytes."""


def write_artifact(raw, *, expected_ref, disease, fold, seed):
    """Compute the artifact manifest from `raw` (the trainer's build output). Hashes are computed from the bytes payloads; any
    hash string present in `raw` is IGNORED (computed, not trusted). Returns a validated artifact manifest."""
    if not isinstance(raw, dict):
        raise Stage1bArtifactWriteError("raw build output must be a dict")
    if raw.get("ref") != expected_ref:
        raise Stage1bArtifactWriteError(f"raw ref {raw.get('ref')!r} != expected {expected_ref!r}")
    art = {"ref": expected_ref, "disease": disease, "fold": fold, "seed": seed}
    for hash_field, bytes_key in HASH_SOURCE.items():
        payload = raw.get(bytes_key)
        if not isinstance(payload, (bytes, bytearray)):
            raise Stage1bArtifactWriteError(f"{expected_ref}: raw['{bytes_key}'] must be bytes (got {type(payload).__name__})")
        art[hash_field] = hashlib.sha256(bytes(payload)).hexdigest()   # COMPUTED from bytes — trainer-reported hashes ignored
    ART.validate_artifact_manifest(art, expected_ref=expected_ref, disease=disease, fold=fold, seed=seed)
    return art
