"""Versioned U1 manifest, handoff, and atomic publication for C85U V2."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from oaci.multidataset.c84s_common import require, sha256_file

from .c85u_result_manifest import publish_utility_field, validate_utility_manifest
from .c85u_runtime_guard_v2 import (
    C85UExecutionContextV2,
    canonical_json_bytes,
    validate_execution_context_v2,
    validate_stage_receipt_v2,
)
from .c85u_u1_registry_v2 import U1RuntimeRegistry


U1_MANIFEST_SCHEMA_V2 = "c85u_complete_utility_manifest_v2"
U1_HANDOFF_SCHEMA_V2 = "c85u_stage_u1_handoff_v2"
U1_MANIFEST_NAME_V2 = "C85U_CANDIDATE_UTILITY_MANIFEST_V2.json"
U1_HANDOFF_NAME = "C85U_STAGE_U1_HANDOFF.json"
MAX_U1_BYTES = 2 * 1024**3


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _write_sidecar(path: Path, digest: str, name: str) -> None:
    with path.open("xb") as handle:
        handle.write(f"{digest}  {name}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def validate_u1_manifest_v2(
    root: str | Path,
    *,
    context: C85UExecutionContextV2 | None = None,
    expected_handoff_sha256: str | None = None,
    expected_contexts: int = 944,
    expected_candidate_rows: int = 76_464,
) -> dict[str, Any]:
    base = Path(root).resolve()
    compatibility = validate_utility_manifest(
        base, expected_contexts=expected_contexts,
        expected_candidate_rows=expected_candidate_rows,
    )
    manifest_path = base / U1_MANIFEST_NAME_V2
    manifest_sidecar = manifest_path.with_suffix(".sha256")
    handoff_path = base / U1_HANDOFF_NAME
    handoff_sidecar = handoff_path.with_suffix(".sha256")
    require(all(path.is_file() for path in (
        manifest_path, manifest_sidecar, handoff_path, handoff_sidecar,
    )), "C85U V2 U1 manifest/handoff absent")
    manifest_sha = sha256_file(manifest_path)
    handoff_sha = sha256_file(handoff_path)
    require(manifest_sidecar.read_text(encoding="ascii").split()
            == [manifest_sha, manifest_path.name],
            "C85U V2 U1 manifest sidecar drift")
    require(handoff_sidecar.read_text(encoding="ascii").split()
            == [handoff_sha, handoff_path.name],
            "C85U V2 U1 handoff sidecar drift")
    require(expected_handoff_sha256 is None or handoff_sha == expected_handoff_sha256,
            "C85U V2 U1 handoff SHA drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == U1_MANIFEST_SCHEMA_V2 and
            manifest.get("status") == "PROVISIONAL_COMPLETE_U1_NOT_ACCEPTED_FOR_C85E",
            "C85U V2 U1 manifest state drift")
    require(handoff.get("schema_version") == U1_HANDOFF_SCHEMA_V2 and
            handoff.get("U1_manifest_sha256") == manifest_sha,
            "C85U V2 U1 handoff linkage drift")
    require(manifest.get("compatibility_manifest_sha256")
            == compatibility["manifest_sha256"],
            "C85U V2 U1 compatibility manifest linkage drift")
    require(manifest.get("contexts") == expected_contexts and
            manifest.get("candidate_rows") == expected_candidate_rows and
            manifest.get("candidates_per_context") == 81,
            "C85U V2 U1 manifest arithmetic drift")
    expected_output = (
        context.output_root / "stage_u1_candidate_utility_v2"
        if context is not None else base
    )
    require(handoff.get("U1_output_root") == str(expected_output),
            "C85U V2 U1 handoff root drift")
    require(int(manifest.get("actual_total_output_bytes", MAX_U1_BYTES + 1)) <= MAX_U1_BYTES,
            "C85U V2 U1 output size exceeds envelope")
    require(int(manifest["actual_total_output_bytes"]) == _tree_bytes(base),
            "C85U V2 U1 recorded output byte total drift")
    forbidden = manifest.get("forbidden_access_counters")
    require(isinstance(forbidden, Mapping) and all(int(value) == 0 for value in forbidden.values()),
            "C85U V2 U1 forbidden-access counter nonzero")
    if context is not None:
        validate_execution_context_v2(context)
        identity = {
            "execution_lock_sha256": context.execution_lock_sha256,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "attempt_id": context.attempt_id,
            "parent_output_root": str(context.output_root),
            "protected_replay_sha256": context.protected_replay_sha256,
        }
        require(all(manifest.get(key) == value for key, value in identity.items()),
                "C85U V2 U1 manifest attempt binding drift")
        require(all(handoff.get(key) == value for key, value in identity.items()),
                "C85U V2 U1 handoff attempt binding drift")
    return {
        "schema_version": U1_MANIFEST_SCHEMA_V2,
        "manifest_sha256": manifest_sha,
        "handoff_sha256": handoff_sha,
        "contexts": int(manifest["contexts"]),
        "candidate_rows": int(manifest["candidate_rows"]),
        "actual_total_output_bytes": int(manifest["actual_total_output_bytes"]),
        "status": "PASS_PROVISIONAL_U1_REPLAY",
    }


def publish_utility_field_v2(
    *, payloads: Iterable[Mapping[str, Any]], final_root: str | Path,
    context: C85UExecutionContextV2, registry: U1RuntimeRegistry,
    stage_receipt_sha256: str, allowed_access_counters: Mapping[str, int],
    forbidden_access_counters: Mapping[str, int],
    expected_contexts: int = 944, expected_candidate_rows: int = 76_464,
) -> dict[str, Any]:
    """Publish V1-compatible artifacts plus authoritative V2 identities atomically."""
    validate_execution_context_v2(context)
    require(context.protected_replay_sha256 is not None,
            "C85U V2 U1 publication precedes protected replay")
    receipt_path, observed_stage_sha = validate_stage_receipt_v2(
        context, "U1", prerequisite_sha256=context.protected_replay_sha256,
    )
    require(observed_stage_sha == stage_receipt_sha256,
            "C85U V2 U1 stage receipt SHA drift")
    require(all(int(value) == 0 for value in forbidden_access_counters.values()),
            "C85U V2 U1 forbidden access occurred")
    final = Path(final_root).resolve()
    require(not final.exists(), "C85U V2 U1 final root exists")
    final.parent.mkdir(parents=True, exist_ok=True)
    building = final.parent / f".{final.name}.v2-building-{uuid4().hex}"
    require(not building.exists(), "C85U V2 U1 building-root collision")
    compatibility = publish_utility_field(
        payloads=payloads,
        final_root=building,
        input_identity={
            "execution_lock_sha256": context.execution_lock_sha256,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "attempt_id": context.attempt_id,
            "protected_input_replay_sha256": context.protected_replay_sha256,
            "evaluation_label_view_manifest_sha256": registry.evaluation_view_manifest_sha256,
            "target_artifacts": len(registry.target_artifact_rows),
            "target_artifact_bytes": registry.target_artifact_total_bytes,
        },
        expected_contexts=expected_contexts,
        expected_candidate_rows=expected_candidate_rows,
    )
    try:
        provisional_bytes = _tree_bytes(building)
        require(provisional_bytes <= MAX_U1_BYTES,
                "C85U V2 U1 output size exceeds envelope before V2 controls")
        manifest = {
            "schema_version": U1_MANIFEST_SCHEMA_V2,
            "status": "PROVISIONAL_COMPLETE_U1_NOT_ACCEPTED_FOR_C85E",
            "execution_lock_sha256": context.execution_lock_sha256,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "authorization_id": context.authorization_id,
            "attempt_id": context.attempt_id,
            "parent_output_root": str(context.output_root),
            "protected_replay_sha256": context.protected_replay_sha256,
            "U1_stage_receipt_sha256": stage_receipt_sha256,
            "U1_stage_receipt_path": str(receipt_path),
            "compatibility_manifest_sha256": compatibility["manifest_sha256"],
            "contexts": expected_contexts,
            "candidates_per_context": 81,
            "candidate_rows": expected_candidate_rows,
            "target_artifacts": len(registry.target_artifact_rows),
            "actual_target_artifact_bytes": registry.target_artifact_total_bytes,
            "target_artifact_registry_sha256": registry.target_artifact_registry_sha256,
            "target_sidecar_registry_sha256": registry.target_sidecar_registry_sha256,
            "evaluation_label_table_rows": registry.evaluation_label_table_rows,
            "evaluation_label_table_sha256": registry.evaluation_label_table_sha256,
            "evaluation_view_manifest_sha256": registry.evaluation_view_manifest_sha256,
            "allowed_access_counters": dict(allowed_access_counters),
            "forbidden_access_counters": dict(forbidden_access_counters),
            "actual_total_output_bytes": 0,
            "output_limit_bytes": MAX_U1_BYTES,
            "accepted_for_C85E": False,
        }
        # The final byte count includes this manifest, its sidecar, and the handoff.
        # Iterate once using a placeholder, then freeze the actual complete total.
        manifest_path = building / U1_MANIFEST_NAME_V2
        manifest["actual_total_output_bytes"] = provisional_bytes
        manifest_sha = _write_json_exclusive(manifest_path, manifest)
        _write_sidecar(manifest_path.with_suffix(".sha256"), manifest_sha, manifest_path.name)
        handoff = {
            "schema_version": U1_HANDOFF_SCHEMA_V2,
            "execution_lock_sha256": context.execution_lock_sha256,
            "execution_lock_commit": context.execution_lock_commit,
            "authorization_file_sha256": context.authorization_file_sha256,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "authorization_id": context.authorization_id,
            "attempt_id": context.attempt_id,
            "parent_output_root": str(context.output_root),
            "U1_output_root": str(final),
            "receipt_path": str(context.protected_replay_path),
            "receipt_sha256": context.protected_replay_sha256,
            "protected_replay_sha256": context.protected_replay_sha256,
            "U1_stage_receipt_path": str(receipt_path),
            "U1_stage_receipt_sha256": stage_receipt_sha256,
            "evaluation_label_table_sha256": registry.evaluation_label_table_sha256,
            "evaluation_label_table_rows": registry.evaluation_label_table_rows,
            "evaluation_view_manifest_sha256": registry.evaluation_view_manifest_sha256,
            "target_artifact_rows": len(registry.target_artifact_rows),
            "target_artifact_total_bytes": registry.target_artifact_total_bytes,
            "target_artifact_registry_sha256": registry.target_artifact_registry_sha256,
            "target_sidecar_registry_sha256": registry.target_sidecar_registry_sha256,
            "U1_manifest_sha256": manifest_sha,
            "accepted_for_C85E": False,
        }
        handoff_path = building / U1_HANDOFF_NAME
        handoff_sha = _write_json_exclusive(handoff_path, handoff)
        _write_sidecar(handoff_path.with_suffix(".sha256"), handoff_sha, handoff_path.name)
        complete_bytes = _tree_bytes(building)
        require(complete_bytes <= MAX_U1_BYTES,
                "C85U V2 U1 complete output exceeds envelope")
        # Re-freeze the manifest once with the complete tree size. The manifest is
        # not yet published, so this is still inside the U1 transaction.
        manifest["actual_total_output_bytes"] = complete_bytes
        manifest_path.unlink()
        manifest_path.with_suffix(".sha256").unlink()
        manifest_sha = _write_json_exclusive(manifest_path, manifest)
        _write_sidecar(manifest_path.with_suffix(".sha256"), manifest_sha, manifest_path.name)
        handoff["U1_manifest_sha256"] = manifest_sha
        handoff_path.unlink()
        handoff_path.with_suffix(".sha256").unlink()
        handoff_sha = _write_json_exclusive(handoff_path, handoff)
        _write_sidecar(handoff_path.with_suffix(".sha256"), handoff_sha, handoff_path.name)
        replay = validate_u1_manifest_v2(
            building, context=context, expected_handoff_sha256=handoff_sha,
            expected_contexts=expected_contexts,
            expected_candidate_rows=expected_candidate_rows,
        )
        for directory in sorted(
            (path for path in building.rglob("*") if path.is_dir()), reverse=True,
        ):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(building, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(building, final)
        return {**replay, "root": str(final)}
    except BaseException:
        raise


__all__ = [
    "MAX_U1_BYTES",
    "U1_HANDOFF_NAME",
    "U1_HANDOFF_SCHEMA_V2",
    "U1_MANIFEST_NAME_V2",
    "U1_MANIFEST_SCHEMA_V2",
    "publish_utility_field_v2",
    "validate_u1_manifest_v2",
]
