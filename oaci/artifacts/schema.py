"""Versioned artifact schema constants + the per-file JSON envelope.

Every artifact JSON carries (schema_version, artifact_kind, logical_hash) at the top level. An unknown
MAJOR schema version is a hard failure -- there is no best-effort read.
"""
from __future__ import annotations

ARTIFACT_SCHEMA_VERSION = "oaci-artifact-v1"
ARTIFACT_PROFILE = "full-fold"

_MAJOR = "v1"

ALLOWED_METHODS = ("ERM", "OACI", "global_lpc", "uniform")


def schema_major(version: str) -> str:
    # "oaci-artifact-v1" -> "v1"
    return str(version).rsplit("-", 1)[-1]


def check_schema_version(version: str) -> None:
    if schema_major(version) != _MAJOR:
        raise ValueError(f"unknown artifact schema major version: {version!r} (expected {_MAJOR})")


def make_envelope(artifact_kind: str, logical_hash: str, body: dict) -> dict:
    if "schema_version" in body or "artifact_kind" in body or "logical_hash" in body:
        raise ValueError("artifact body must not set reserved envelope keys")
    return {"schema_version": ARTIFACT_SCHEMA_VERSION, "artifact_kind": str(artifact_kind),
            "logical_hash": str(logical_hash), "body": body}


def open_envelope(doc: dict, expected_kind: str):
    check_schema_version(doc.get("schema_version", ""))
    if doc.get("artifact_kind") != expected_kind:
        raise ValueError(f"artifact kind mismatch: got {doc.get('artifact_kind')!r}, expected {expected_kind!r}")
    return doc["logical_hash"], doc["body"]
