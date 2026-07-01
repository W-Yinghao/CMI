"""ACAR V5 Stage-1B FILE-backed artifact writer (stdlib). For a REAL build the trainer/dumper emit large output FILES, not in-memory
bytes; this writer streams sha256 over each file (rejecting missing/empty), IGNORES any trainer-reported hash, and emits the same
validated 6-field registry manifest. It also enforces OUTPUT-LAYOUT containment: when a run_id is supplied (real runs), every
artifact file must be a non-symlink regular file under  output_root/run_id/safe_ref_slug(ref)/  (per-ref containment); when only
output_root is supplied (legacy synthetic tests), it enforces plain output_root containment. Synthetic-tested with temp files (no
DEV, no model).
"""
from __future__ import annotations
import hashlib
import os
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_artifacts as ART
from acar.v5.substrate import stage1b_output_layout as LO

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
    """Raised when a trainer's file outputs are missing / empty / mis-specified / uncontained."""


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


def write_artifact_from_files(raw, *, expected_ref, disease, fold, seed, output_root=None, run_id=None):
    """Compute the artifact manifest by streaming sha256 over the trainer's output files. Any hash string in `raw` is IGNORED.
    Containment:
      - run_id given (real runs): EVERY artifact file must be a non-symlink regular file under output_root/run_id/safe_ref_slug(ref);
      - only output_root given (legacy): every file must realpath-resolve inside output_root and not be a symlink;
    and in all cases the 6 files must be distinct (no path reused across artifacts). The manifest also records the resolved paths
    (under key '_paths', ref-local) so the finalize barrier can enforce GLOBAL cross-ref uniqueness + config-sidecar validation."""
    if not isinstance(raw, dict):
        raise Stage1bFileArtifactError("raw build output must be a dict")
    if raw.get("ref") != expected_ref:
        raise Stage1bFileArtifactError(f"raw ref {raw.get('ref')!r} != expected {expected_ref!r}")
    root_real = os.path.realpath(output_root) + os.sep if output_root else None
    art = {"ref": expected_ref, "disease": disease, "fold": fold, "seed": seed}
    seen_paths, paths = {}, {}
    for hash_field, path_key in FILE_SOURCE.items():
        path = raw.get(path_key)
        if not isinstance(path, str) or not path:
            raise Stage1bFileArtifactError(f"{expected_ref}: raw['{path_key}'] must be a non-empty path")
        if os.path.islink(path):
            raise Stage1bFileArtifactError(f"{expected_ref}: artifact path is a symlink (rejected): {path}")
        if not os.path.isfile(path):
            raise Stage1bFileArtifactError(f"{expected_ref}: artifact file missing: {path}")
        try:
            if run_id is not None:                            # real-run per-ref containment (also rejects symlink components)
                real = LO.assert_ref_file_contained(path, output_root, run_id, expected_ref)
            elif root_real is not None:                       # legacy plain output_root containment
                real = os.path.realpath(path)
                if not (real + os.sep).startswith(root_real) and real != root_real.rstrip(os.sep):
                    raise Stage1bFileArtifactError(f"{expected_ref}: artifact {path} escapes output_root {output_root}")
            else:
                real = os.path.realpath(path)
        except LO.Stage1bLayoutError as e:
            raise Stage1bFileArtifactError(f"{expected_ref}: {e}")
        if real in seen_paths:
            raise Stage1bFileArtifactError(f"{expected_ref}: duplicate artifact path shared by {seen_paths[real]} and {path_key}")
        seen_paths[real] = path_key
        paths[path_key] = path
        art[hash_field] = _sha256_stream(path)                # streamed from the actual file bytes (computed, not trusted)
    ART.validate_artifact_manifest(art, expected_ref=expected_ref, disease=disease, fold=fold, seed=seed)
    art["_paths"] = paths                                     # ref-local resolved paths (finalize enforces global uniqueness)
    return art
