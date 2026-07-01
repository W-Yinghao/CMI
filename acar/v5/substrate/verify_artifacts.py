"""ACAR V5 artifact verification — file-byte SHA-256 provenance check (stdlib only; NO torch, NO model load, NO real cohort read).
This is the Step-3 analogue of the v4 preflight's file-byte hashing: it proves an on-disk artifact matches a pinned hash WITHOUT
importing/executing it. Reading model weights or DEV signal is out of scope for the synthetic scaffold.
"""
from __future__ import annotations
import hashlib
import os

_HEX = "0123456789abcdef"


class ArtifactHashMismatch(RuntimeError):
    """Raised when an on-disk artifact's byte-sha does not match its pinned value."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(ch in _HEX for ch in s)


def verify_artifact_file(path, expected_sha256):
    """Fail-closed: the file must EXIST and its byte-sha must equal `expected_sha256` (64-hex). Returns the sha on success."""
    if not _is_hex64(expected_sha256):
        raise ValueError("expected_sha256 must be 64-char lowercase hex")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"artifact missing: {path}")
    got = sha256_file(path)
    if got != expected_sha256:
        raise ArtifactHashMismatch(f"{path}: sha256 {got} != expected {expected_sha256}")
    return got


def verify_registry_entry_files(file_map):
    """Verify a {path: expected_sha256} map; raises on the first mismatch/missing. Returns the dict of verified shas."""
    return {p: verify_artifact_file(p, s) for p, s in file_map.items()}
