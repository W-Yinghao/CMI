"""Atomic C85T result publication and artifact identity replay."""
from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from .c85_decision_experiments import DecisionContractError


RESULT_SCHEMA = "c85t_decision_theory_proof_and_synthetic_result_v1"
MANIFEST_SCHEMA = "c85t_result_artifact_manifest_v1"
ATTEMPT_SCHEMA = "c85t_execution_attempt_ledger_v1"
SUCCESS_GATE = (
    "C85T_DECISION_THEORY_PROOF_AUDIT_AND_SYNTHETIC_VALIDATION_COMPLETE_"
    "C85E_PROTOCOL_REVIEW_REQUIRED"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_attempt_ledger(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise DecisionContractError("C85T attempt ledger already exists")
    payload = {"schema_version": ATTEMPT_SCHEMA, **value}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


class AtomicResultWriter(AbstractContextManager["AtomicResultWriter"]):
    """Write a complete result in staging and publish it with one rename."""

    def __init__(self, output_root: Path, *, failure_injection: str | None = None):
        self.output_root = output_root.resolve()
        self.staging_root = self.output_root.with_name(
            f".{self.output_root.name}.staging-{uuid4().hex}"
        )
        self.failure_injection = failure_injection
        self._published = False
        self.failed_root: Path | None = None

    def __enter__(self) -> "AtomicResultWriter":
        if self.output_root.exists():
            raise DecisionContractError("C85T output root must be absent")
        self.staging_root.mkdir(parents=True, exist_ok=False)
        return self

    def path(self, relative: str | Path) -> Path:
        path = (self.staging_root / relative).resolve()
        if self.staging_root not in path.parents and path != self.staging_root:
            raise DecisionContractError("result path escapes staging root")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str | Path, value: Any) -> Path:
        path = self.path(relative)
        path.write_bytes(canonical_json_bytes(value))
        return path

    def write_text(self, relative: str | Path, value: str) -> Path:
        path = self.path(relative)
        path.write_text(value)
        return path

    def _artifact_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.staging_root.rglob("*")):
            if not path.is_file() or path.name == "C85T_RESULT_ARTIFACT_MANIFEST.json":
                continue
            rows.append(
                {
                    "path": str(path.relative_to(self.staging_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return rows

    def publish(self, result: dict[str, Any]) -> Path:
        if self._published:
            raise DecisionContractError("C85T staging root was already published")
        if result.get("schema_version") != RESULT_SCHEMA:
            raise DecisionContractError("C85T result schema drifted")
        if result.get("final_gate") != SUCCESS_GATE:
            raise DecisionContractError("C85T success gate drifted")
        if self.failure_injection == "before_result":
            raise RuntimeError("C85T_SHADOW_FAILURE_BEFORE_RESULT")
        self.write_json("C85T_RESULT.json", result)
        rows = self._artifact_rows()
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "created_at_utc": utc_now(),
            "artifact_count": len(rows),
            "artifacts": rows,
        }
        if self.failure_injection == "before_manifest":
            raise RuntimeError("C85T_SHADOW_FAILURE_BEFORE_MANIFEST")
        self.write_json("C85T_RESULT_ARTIFACT_MANIFEST.json", manifest)
        replay_manifest(self.staging_root)
        if self.failure_injection == "before_publish":
            raise RuntimeError("C85T_SHADOW_FAILURE_BEFORE_PUBLISH")
        os.replace(self.staging_root, self.output_root)
        self._published = True
        return self.output_root

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if not self._published and self.staging_root.exists():
            if exc_type is None:
                shutil.rmtree(self.staging_root)
            else:
                self.failed_root = self.output_root.with_name(
                    f"{self.output_root.name}.failed-{uuid4().hex}"
                )
                os.replace(self.staging_root, self.failed_root)
        return False


def replay_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "C85T_RESULT_ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        raise DecisionContractError("C85T artifact manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise DecisionContractError("C85T artifact manifest schema drifted")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise DecisionContractError("C85T artifact manifest count drifted")
    observed_paths: set[str] = set()
    for row in rows:
        relative = row["path"]
        if relative in observed_paths:
            raise DecisionContractError("duplicate C85T artifact path")
        observed_paths.add(relative)
        path = root / relative
        if not path.is_file():
            raise DecisionContractError(f"C85T artifact is absent: {relative}")
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise DecisionContractError(f"C85T artifact identity drift: {relative}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest_path.name
    }
    if actual != observed_paths:
        raise DecisionContractError("C85T artifact manifest coverage drifted")
    return manifest
