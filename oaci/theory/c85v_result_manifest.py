"""Atomic C85V result bundle and semantic replay."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .c85_decision_experiments import DecisionContractError
from .c85v_adjudication import ALLOWED_STATUSES, replay_adjudication
from .c85v_statement_registry import (
    THEOREM_IDS,
    canonical_json_bytes,
    sha256_file,
)


BUNDLE_SCHEMA = "c85v_atomic_proof_review_bundle_v1"
MANIFEST_SCHEMA = "c85v_result_artifact_manifest_v1"
RESULT_SCHEMA = "c85v_independent_proof_verdict_result_v1"
COMPLETION_SCHEMA = "c85v_completion_receipt_v1"
SUCCESS_GATE = (
    "C85V_INDEPENDENT_PROOF_VERDICTS_AND_THEOREM_STATUSES_FROZEN_"
    "C85E_PROTOCOL_REVIEW_REQUIRED"
)
LIFECYCLE_EVENTS = (
    "PREFLIGHT_STARTED",
    "PREFLIGHT_COMPLETED",
    "AUTHORIZATION_CONSUMED",
    "STAGE_A_STARTED",
    "STAGE_A_COMPLETED",
    "STAGE_B_STARTED",
    "STAGE_B_COMPLETED",
    "ADJUDICATION_STARTED",
    "ADJUDICATION_COMPLETED",
    "MANIFEST_COMPLETED",
    "ATOMIC_PUBLISH_COMMIT_READY",
)
CONTROL_FILES = {
    "C85V_RESULT_ARTIFACT_MANIFEST.json",
    "C85V_LIFECYCLE.jsonl",
    "C85V_COMPLETION_RECEIPT.json",
}


def _lifecycle_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            raise DecisionContractError("C85V lifecycle contains an empty line")
        rows.append(json.loads(line))
    if [row.get("stage") for row in rows] != list(LIFECYCLE_EVENTS[: len(rows)]):
        raise DecisionContractError("C85V lifecycle order drifted")
    if [row.get("sequence") for row in rows] != list(range(len(rows))):
        raise DecisionContractError("C85V lifecycle sequence drifted")
    return rows


def _lifecycle_prefix_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in CONTROL_FILES:
            continue
        rows.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DecisionContractError("C85V result bundle cannot contain symlinks")
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            directories.append(path)
    for path in reversed(directories + [root]):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def replay_artifact_manifest(root: Path) -> dict[str, Any]:
    path = root / "C85V_RESULT_ARTIFACT_MANIFEST.json"
    if not path.is_file():
        raise DecisionContractError("C85V result manifest is absent")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise DecisionContractError("C85V result manifest schema drifted")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise DecisionContractError("C85V result manifest count drifted")
    observed: set[str] = set()
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str) or relative in observed or relative in CONTROL_FILES:
            raise DecisionContractError("C85V result manifest path drifted")
        observed.add(relative)
        artifact = root / relative
        if (
            not artifact.is_file()
            or artifact.stat().st_size != row.get("size_bytes")
            or sha256_file(artifact) != row.get("sha256")
        ):
            raise DecisionContractError(f"C85V result artifact drifted: {relative}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and str(path.relative_to(root)) not in CONTROL_FILES
    }
    if observed != actual:
        raise DecisionContractError("C85V result manifest coverage drifted")
    return manifest


def validate_complete_bundle(
    root: Path,
    *,
    expected_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    required = {
        "authorization_consumed.json",
        "primary_literature_registry.csv",
        "proof_candidate_retention_ledger.csv",
        "C85V_RESULT.json",
        "stage_a/C85V_STAGE_A_DERIVATION_MANIFEST.json",
        "stage_a/C85V_STAGE_A_DERIVATION_MANIFEST.sha256",
        "stage_b/C85V_STAGE_B_COMPARISON_MANIFEST.json",
        "stage_b/C85V_STAGE_B_COMPARISON_MANIFEST.sha256",
        "adjudication/C85V_ADJUDICATION_MANIFEST.json",
        "adjudication/C85V_ADJUDICATION_MANIFEST.sha256",
        "adjudication/formal_theorem_status_registry.csv",
    }
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if not required.issubset(actual_files):
        missing = sorted(required - actual_files)
        raise DecisionContractError(f"C85V required result object is absent: {missing[0]}")
    manifest = replay_artifact_manifest(root)
    result_path = root / "C85V_RESULT.json"
    if not result_path.is_file():
        raise DecisionContractError("C85V result JSON is absent")
    result = json.loads(result_path.read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("final_gate") != SUCCESS_GATE
        or result.get("theorem_count") != 7
        or result.get("monte_carlo_reruns") != 0
        or result.get("real_data_access") != 0
        or result.get("active_acquisition") != 0
        or result.get("C85E_authorized") is not False
        or result.get("manuscript_modified") is not False
    ):
        raise DecisionContractError("C85V protected result contract drifted")
    statuses = result.get("formal_theorem_statuses")
    if not isinstance(statuses, dict) or set(statuses) != set(THEOREM_IDS):
        raise DecisionContractError("C85V theorem-status coverage drifted")
    if any(statuses[key] not in ALLOWED_STATUSES[key] for key in THEOREM_IDS):
        raise DecisionContractError("C85V theorem-status value drifted")
    review_mode = result.get("review_mode")
    if review_mode not in {"REGISTERED_C85V", "SHADOW_C85VP"}:
        raise DecisionContractError("C85V result review mode drifted")
    replay_adjudication(
        root / "adjudication",
        stage_a_root=root / "stage_a",
        stage_b_root=root / "stage_b",
        expected_review_mode=review_mode,
    )
    retention_path = root / "proof_candidate_retention_ledger.csv"
    with retention_path.open(newline="") as handle:
        retention = list(csv.DictReader(handle))
    if (
        len(retention) != 7
        or {row.get("theorem_id") for row in retention} != set(THEOREM_IDS)
        or any(row.get("overwritten") != "0" for row in retention)
    ):
        raise DecisionContractError("C85V proof-candidate retention ledger drifted")
    if any(path.suffix == ".npz" for path in root.rglob("*")):
        raise DecisionContractError("C85V result illegally contains Monte Carlo arrays")
    lifecycle_path = root / "C85V_LIFECYCLE.jsonl"
    events = _lifecycle_lines(lifecycle_path)
    if [row["stage"] for row in events] != list(LIFECYCLE_EVENTS):
        raise DecisionContractError("C85V success lifecycle is incomplete")
    completion_path = root / "C85V_COMPLETION_RECEIPT.json"
    completion = json.loads(completion_path.read_text())
    lifecycle_lines = lifecycle_path.read_bytes().splitlines(keepends=True)
    lifecycle_prefix_sha = hashlib.sha256(b"".join(lifecycle_lines[:-1])).hexdigest()
    if (
        completion.get("schema_version") != COMPLETION_SCHEMA
        or completion.get("manifest_sha256") != sha256_file(
            root / "C85V_RESULT_ARTIFACT_MANIFEST.json"
        )
        or completion.get("lifecycle_prefix_sha256") != lifecycle_prefix_sha
        or events[-1].get("artifact_or_receipt_sha256") != sha256_file(completion_path)
    ):
        raise DecisionContractError("C85V completion receipt chain drifted")
    if expected_identity:
        for key, value in expected_identity.items():
            if result.get(key) != value or completion.get(key) != value:
                raise DecisionContractError(f"C85V result identity drifted: {key}")
    if manifest.get("derived_counts") != {
        "stage_a_derivations": 7,
        "stage_b_comparisons": 7,
        "adversarial_audits": 7,
        "final_verdicts": 7,
        "formal_status_rows": 7,
        "monte_carlo_reruns": 0,
    }:
        raise DecisionContractError("C85V derived result counts drifted")
    return result


class AtomicC85VResultBundle:
    """Prepare a complete review bundle and publish it with one final rename."""

    def __init__(
        self,
        *,
        output_root: Path,
        attempt_id: str,
        identity: Mapping[str, str],
        review_mode: str = "REGISTERED_C85V",
    ) -> None:
        self.output_root = output_root.resolve()
        self.staging_root = self.output_root.with_name(
            f".{self.output_root.name}.staging-{attempt_id}"
        )
        if self.output_root.exists() or self.staging_root.exists():
            raise DecisionContractError("C85V output and staging roots must be fresh")
        self.staging_root.mkdir(parents=True)
        self.identity = dict(identity)
        self.attempt_id = attempt_id
        if review_mode not in {"REGISTERED_C85V", "SHADOW_C85VP"}:
            raise DecisionContractError("C85V atomic bundle review mode is invalid")
        self.review_mode = review_mode
        self.lifecycle_path = self.staging_root / "C85V_LIFECYCLE.jsonl"

    def append_event(self, stage: str, artifact_or_receipt_sha256: str = "") -> None:
        rows = _lifecycle_lines(self.lifecycle_path) if self.lifecycle_path.exists() else []
        expected = LIFECYCLE_EVENTS[len(rows)] if len(rows) < len(LIFECYCLE_EVENTS) else None
        if stage != expected:
            raise DecisionContractError("C85V lifecycle event order drifted")
        row = {
            "schema_version": "c85v_lifecycle_event_v1",
            "sequence": len(rows),
            "stage": stage,
            "attempt_id": self.attempt_id,
            **self.identity,
            "artifact_or_receipt_sha256": artifact_or_receipt_sha256,
        }
        with self.lifecycle_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    def write_json(self, relative: str, value: Any) -> Path:
        path = self.staging_root / relative
        if path.exists():
            raise DecisionContractError("C85V result artifact must be fresh")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value))
        return path

    def prepare_and_publish(self, result: Mapping[str, Any]) -> dict[str, Any]:
        result_value = dict(result)
        result_value["review_mode"] = self.review_mode
        self.write_json("C85V_RESULT.json", result_value)
        rows = _artifact_rows(self.staging_root)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "bundle_schema": BUNDLE_SCHEMA,
            "artifact_count": len(rows),
            "artifacts": rows,
            "derived_counts": {
                "stage_a_derivations": 7,
                "stage_b_comparisons": 7,
                "adversarial_audits": 7,
                "final_verdicts": 7,
                "formal_status_rows": 7,
                "monte_carlo_reruns": 0,
            },
        }
        manifest_path = self.write_json("C85V_RESULT_ARTIFACT_MANIFEST.json", manifest)
        self.append_event("MANIFEST_COMPLETED", sha256_file(manifest_path))
        completion = {
            "schema_version": COMPLETION_SCHEMA,
            "bundle_schema": BUNDLE_SCHEMA,
            "attempt_id": self.attempt_id,
            **self.identity,
            "manifest_sha256": sha256_file(manifest_path),
            "lifecycle_prefix_sha256": _lifecycle_prefix_sha(self.lifecycle_path),
            "final_gate": SUCCESS_GATE,
        }
        completion_path = self.write_json("C85V_COMPLETION_RECEIPT.json", completion)
        self.append_event("ATOMIC_PUBLISH_COMMIT_READY", sha256_file(completion_path))
        validate_complete_bundle(self.staging_root, expected_identity=self.identity)
        _fsync_tree(self.staging_root)
        os.replace(self.staging_root, self.output_root)
        return dict(completion)
