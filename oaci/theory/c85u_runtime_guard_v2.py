"""Authorization, protected replay, and stage-attempt guards for C85U V2."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
from uuid import UUID, uuid4

import numpy as np
import scipy

from oaci.multidataset.c84s_common import canonical_sha256, require, sha256_file

from .c85u_u1_registry_v2 import (
    EVALUATION_VIEW_MANIFEST,
    EXPECTED_TARGET_ARTIFACT_BYTES,
    U1RuntimeRegistry,
)


AUTHORIZATION_SCHEMA_V2 = "c85u_direct_pi_authorization_record_v2"
CONSUMPTION_SCHEMA_V2 = "c85u_authorization_consumption_receipt_v2"
CONTEXT_SCHEMA_V2 = "c85u_execution_context_receipt_v2"
PROTECTED_REPLAY_SCHEMA_V2 = "c85u_protected_input_replay_receipt_v2"
STAGE_RECEIPT_SCHEMA_V2 = "c85u_stage_exclusive_receipt_v2"
LIFECYCLE_SCHEMA_V2 = "c85u_append_only_lifecycle_v2"
LOCK_SCHEMA_V2 = "c85u_execution_lock_v2"
LOCK_STATUS_V2 = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
LOCK_PATH_NAME_V2 = "C85U_EXECUTION_LOCK_V2.json"
AUTHORIZATION_PATH_NAME_V2 = "C85U_V2_PI_AUTHORIZATION_RECORD.json"
DIRECT_STATEMENT = "授权 C85U"
PROTECTED_FALSE_FIELDS = (
    "C85E", "C86", "active_acquisition", "real_data",
    "new_data_or_model_zoo", "manuscript",
)
SUCCESS_LIFECYCLE_V2 = (
    "PREFLIGHT_STARTED",
    "PREFLIGHT_COMPLETED",
    "AUTHORIZATION_CONSUMED",
    "PROTECTED_INPUT_REPLAY_STARTED",
    "PROTECTED_INPUT_REPLAY_COMPLETED",
    "STAGE_U1_STARTED",
    "STAGE_U1_COMPLETED",
    "STAGE_U2_STARTED",
    "STAGE_U2_COMPLETED",
    "ACCEPTANCE_MANIFEST_STARTED",
    "ACCEPTANCE_MANIFEST_COMPLETED",
    "ATOMIC_ACCEPTANCE_COMMIT_READY",
)
TERMINAL_LIFECYCLE_EVENTS = {"ATOMIC_ACCEPTANCE_COMMIT_READY", "FAILED"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONSUMPTION_PATH_MARKER = "C85U_V2_DERIVED_CONSUMPTION_PATH"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root, check=False, capture_output=True,
    ).returncode == 0


def _valid_authorization_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if _HEX64.fullmatch(value):
        return True
    try:
        UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def authorization_binding_sha256(record: Mapping[str, Any]) -> str:
    normalized = dict(record)
    normalized["consumption_ledger_path"] = _CONSUMPTION_PATH_MARKER
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def expected_output_root(
    parent: str | Path, lock_sha256: str, authorization_id: str,
) -> Path:
    require(_HEX64.fullmatch(lock_sha256) is not None,
            "C85U V2 invalid execution-lock SHA")
    require(_valid_authorization_id(authorization_id),
            "C85U V2 invalid authorization ID")
    compact = authorization_id.replace("-", "").lower()[:16]
    return Path(parent).resolve() / f"c85u-v2-{lock_sha256[:16]}-{compact}"


def expected_consumption_path(root: str | Path, binding_sha256: str) -> Path:
    require(_HEX64.fullmatch(binding_sha256) is not None,
            "C85U V2 invalid authorization binding SHA")
    return Path(root).resolve() / f"{binding_sha256}.json"


def _write_exclusive_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"C85U V2 exclusive receipt already exists: {path.name}") from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "C85U V2 receipt short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def replay_execution_lock_v2(
    lock_path: str | Path,
) -> tuple[dict[str, Any], str, Path, str]:
    path = Path(lock_path).resolve()
    sidecar = path.with_suffix(".sha256")
    require(path.name == LOCK_PATH_NAME_V2 and path.parent.name == "reports" and
            path.parent.parent.name == "oaci", "C85U V2 lock path drift")
    repo_root = path.parents[2]
    require(path.is_file() and sidecar.is_file(), "C85U V2 lock or sidecar absent")
    digest = sha256_file(path)
    require(sidecar.read_text(encoding="ascii").split() == [digest, path.name],
            "C85U V2 lock sidecar drift")
    lock = json.loads(path.read_text(encoding="utf-8"))
    require(lock.get("schema_version") == LOCK_SCHEMA_V2 and
            lock.get("status") == LOCK_STATUS_V2 and lock.get("authorized") is False,
            "C85U V2 lock state drift")
    require(Path(str(lock.get("repo_root", ""))).resolve() == repo_root,
            "C85U V2 lock repository drift")
    bound = lock.get("bound_repository_objects")
    require(isinstance(bound, list) and len(bound) == lock.get("runtime_bound_object_count"),
            "C85U V2 bound-object registry drift")
    seen: set[str] = set()
    for row in bound:
        relative = str(row["path"])
        require(relative not in seen, "C85U V2 duplicate bound object")
        seen.add(relative)
        candidate = repo_root / relative
        require(candidate.is_file() and candidate.stat().st_size == int(row["size_bytes"]),
                f"C85U V2 bound object path/size drift: {relative}")
        require(sha256_file(candidate) == str(row["sha256"]) and
                _git(repo_root, "hash-object", "--", relative) == str(row["git_blob"]),
                f"C85U V2 bound object byte/Git drift: {relative}")
    registry = lock.get("runtime_bound_registry")
    require(isinstance(registry, Mapping), "C85U V2 runtime-bound registry absent")
    registry_path = repo_root / str(registry["path"])
    require(registry_path.is_file() and
            registry_path.stat().st_size == int(registry["size_bytes"]) and
            sha256_file(registry_path) == str(registry["sha256"]) and
            _git(repo_root, "hash-object", "--", str(registry["path"]))
            == str(registry["git_blob"]), "C85U V2 runtime-bound registry drift")
    relative_lock = str(path.relative_to(repo_root))
    lock_commit = _git(repo_root, "log", "-1", "--format=%H", "--", relative_lock)
    require(bool(lock_commit), "C85U V2 lock is not committed")
    return lock, digest, repo_root, lock_commit


def _replay_repository_state(
    repo_root: Path, lock: Mapping[str, Any], lock_commit: str,
) -> str:
    require(_git(repo_root, "branch", "--show-current") == "oaci",
            "C85U V2 requires branch oaci")
    require(not _git(repo_root, "status", "--porcelain"),
            "C85U V2 requires clean worktree")
    head = _git(repo_root, "rev-parse", "HEAD")
    require(head == _git(repo_root, "rev-parse", "origin/oaci"),
            "C85U V2 requires HEAD == origin/oaci")
    require(_is_ancestor(repo_root, str(lock["implementation_commit"]), lock_commit)
            and _is_ancestor(repo_root, lock_commit, head),
            "C85U V2 implementation/lock/HEAD chronology drift")
    return head


def _replay_environment_and_storage(lock: Mapping[str, Any]) -> None:
    environment = lock["environment"]
    require(str(Path(sys.executable).resolve()) == str(environment["python_executable"]),
            "C85U V2 Python executable drift")
    require(sys.version.split()[0] == str(environment["python_version"]),
            "C85U V2 Python version drift")
    require(np.__version__ == str(environment["numpy_version"]) and
            sha256_file(Path(np.__file__)) == str(environment["numpy_file_sha256"]),
            "C85U V2 NumPy identity drift")
    require(scipy.__version__ == str(environment["scipy_version"]) and
            sha256_file(Path(scipy.__file__)) == str(environment["scipy_file_sha256"]),
            "C85U V2 SciPy identity drift")
    parent = Path(str(lock["output_root_policy"]["parent"])).resolve()
    existing = parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    require(shutil.disk_usage(existing).free >= int(lock["output_root_policy"]["max_bytes"]),
            "C85U V2 output filesystem lacks locked storage")


class AppendOnlyLifecycleV2:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def replay(self) -> list[dict[str, Any]]:
        require(self.path.is_file(), "C85U V2 lifecycle absent")
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        require(rows, "C85U V2 lifecycle empty")
        require([int(row["sequence"]) for row in rows] == list(range(len(rows))),
                "C85U V2 lifecycle sequence drift")
        return rows

    def append(
        self, stage: str, *, context: "C85UExecutionContextV2",
        artifact_or_receipt_sha256: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows = self.replay()
        require(rows[-1]["stage"] not in TERMINAL_LIFECYCLE_EVENTS,
                "C85U V2 lifecycle already terminal")
        expected_prefix = list(SUCCESS_LIFECYCLE_V2[: len(rows)])
        require([row["stage"] for row in rows] == expected_prefix,
                "C85U V2 lifecycle prefix drift")
        require(len(rows) < len(SUCCESS_LIFECYCLE_V2) and
                stage == SUCCESS_LIFECYCLE_V2[len(rows)],
                "C85U V2 lifecycle transition drift")
        row = {
            "schema_version": LIFECYCLE_SCHEMA_V2,
            "sequence": len(rows),
            "timestamp_unix_ns": time.time_ns(),
            "stage": stage,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "execution_lock_sha256": context.execution_lock_sha256,
            "attempt_id": context.attempt_id,
            "output_root": str(context.output_root),
            "artifact_or_receipt_sha256": artifact_or_receipt_sha256,
            "details": dict(details or {}),
        }
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def append_failed(
        self, *, context: "C85UExecutionContextV2", primary: BaseException,
        secondary_errors: list[str],
    ) -> bool:
        rows = self.replay()
        if rows[-1]["stage"] in TERMINAL_LIFECYCLE_EVENTS:
            return False
        row = {
            "schema_version": LIFECYCLE_SCHEMA_V2,
            "sequence": len(rows),
            "timestamp_unix_ns": time.time_ns(),
            "stage": "FAILED",
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "execution_lock_sha256": context.execution_lock_sha256,
            "attempt_id": context.attempt_id,
            "output_root": str(context.output_root),
            "last_completed_stage": rows[-1]["stage"],
            "primary_exception_type": type(primary).__name__,
            "primary_exception_message": str(primary),
            "secondary_errors": list(secondary_errors),
        }
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        return True


def _create_lifecycle(
    path: Path, *, binding_sha: str, lock_sha: str, attempt_id: str, output_root: Path,
) -> AppendOnlyLifecycleV2:
    path.parent.mkdir(parents=True, exist_ok=False)
    initial = {
        "schema_version": LIFECYCLE_SCHEMA_V2,
        "sequence": 0,
        "timestamp_unix_ns": time.time_ns(),
        "stage": "PREFLIGHT_STARTED",
        "authorization_binding_sha256": binding_sha,
        "execution_lock_sha256": lock_sha,
        "attempt_id": attempt_id,
        "output_root": str(output_root),
        "artifact_or_receipt_sha256": None,
        "details": {},
    }
    _write_exclusive_fsynced(path, canonical_json_bytes(initial))
    return AppendOnlyLifecycleV2(path)


@dataclass(frozen=True)
class C85UExecutionContextV2:
    authorization_file_sha256: str
    authorization_binding_sha256: str
    authorization_id: str
    execution_lock_path: Path
    execution_lock_sha256: str
    execution_lock_commit: str
    attempt_id: str
    output_root: Path
    receipt_path: Path
    receipt_sha256: str
    head: str
    lifecycle_path: Path
    acceptance_staging_root: Path
    protected_replay_path: Path | None = None
    protected_replay_sha256: str | None = None


def _context_identity(context: C85UExecutionContextV2) -> dict[str, Any]:
    return {
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "execution_lock_path": str(context.execution_lock_path),
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "HEAD": context.head,
    }


def validate_execution_context_v2(context: C85UExecutionContextV2) -> None:
    require(isinstance(context, C85UExecutionContextV2),
            "C85U V2 execution context type drift")
    require(context.receipt_path.is_file() and
            sha256_file(context.receipt_path) == context.receipt_sha256,
            "C85U V2 external authorization receipt drift")
    receipt = json.loads(context.receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema_version") == CONSUMPTION_SCHEMA_V2,
            "C85U V2 external receipt schema drift")
    require(all(receipt.get(key) == value for key, value in _context_identity(context).items()),
            "C85U V2 external receipt/context linkage drift")
    rows = AppendOnlyLifecycleV2(context.lifecycle_path).replay()
    require(any(row["stage"] == "AUTHORIZATION_CONSUMED" for row in rows),
            "C85U V2 lifecycle lacks authorization consumption")
    require(all(
        row["authorization_binding_sha256"] == context.authorization_binding_sha256
        and row["execution_lock_sha256"] == context.execution_lock_sha256
        and row["attempt_id"] == context.attempt_id
        and row["output_root"] == str(context.output_root)
        for row in rows
    ), "C85U V2 lifecycle/context linkage drift")


def validate_context_lock_v2(context: C85UExecutionContextV2) -> dict[str, Any]:
    """Replay the committed V2 lock and bind it to an existing attempt context."""
    lock, digest, repo_root, commit = replay_execution_lock_v2(context.execution_lock_path)
    require(digest == context.execution_lock_sha256 and
            commit == context.execution_lock_commit,
            "C85U V2 execution context/committed lock drift")
    require(_replay_repository_state(repo_root, lock, commit) == context.head,
            "C85U V2 execution context/repository HEAD drift")
    return lock


def _validate_authorization(
    *, path: Path, output_root: Path, lock: Mapping[str, Any], lock_sha: str,
    lock_commit: str, repo_root: Path,
) -> tuple[dict[str, Any], str, str]:
    canonical = repo_root / "oaci/reports" / AUTHORIZATION_PATH_NAME_V2
    require(path.resolve() == canonical and path.is_file(),
            "C85U V2 authorization must use committed canonical path")
    relative = str(canonical.relative_to(repo_root))
    _git(repo_root, "ls-files", "--error-unmatch", relative)
    record = json.loads(canonical.read_text(encoding="utf-8"))
    required = {
        "schema_version": AUTHORIZATION_SCHEMA_V2,
        "direct_explicit_PI_authorization": True,
        "direct_statement_exact": DIRECT_STATEMENT,
        "authorized_stage": "C85U",
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
    }
    require(all(record.get(key) == value for key, value in required.items()),
            "C85U V2 authorization binding drift")
    require(all(record.get(key) is False for key in PROTECTED_FALSE_FIELDS),
            "C85U V2 protected authorization field drift")
    authorization_id = str(record.get("authorization_id", ""))
    require(_valid_authorization_id(authorization_id), "C85U V2 authorization ID malformed")
    expected_root = expected_output_root(
        lock["output_root_policy"]["parent"], lock_sha, authorization_id,
    )
    require(Path(str(record.get("output_root", ""))).resolve() == expected_root and
            output_root.resolve() == expected_root,
            "C85U V2 authorization output-root binding drift")
    binding_sha = authorization_binding_sha256(record)
    expected_receipt = expected_consumption_path(
        lock["authorization_consumption_root"], binding_sha,
    )
    require(Path(str(record.get("consumption_ledger_path", ""))).resolve() == expected_receipt,
            "C85U V2 authorization consumption-path drift")
    authorization_commit = _git(repo_root, "log", "-1", "--format=%H", "--", relative)
    require(_is_ancestor(repo_root, lock_commit, authorization_commit),
            "C85U V2 authorization does not follow lock")
    return record, sha256_file(canonical), binding_sha


def create_execution_context_v2(
    *, execution_lock: str | Path, authorization_record: str | Path,
    output_root: str | Path,
) -> tuple[C85UExecutionContextV2, dict[str, Any]]:
    """Replay preflight, consume one authorization, and create one attempt."""
    lock, lock_sha, repo_root, lock_commit = replay_execution_lock_v2(execution_lock)
    head = _replay_repository_state(repo_root, lock, lock_commit)
    _replay_environment_and_storage(lock)
    output = Path(output_root).resolve()
    record, authorization_sha, binding_sha = _validate_authorization(
        path=Path(authorization_record), output_root=output, lock=lock,
        lock_sha=lock_sha, lock_commit=lock_commit, repo_root=repo_root,
    )
    receipt_path = expected_consumption_path(lock["authorization_consumption_root"], binding_sha)
    require(not output.exists(), "C85U V2 authorization-bound output root exists")
    require(not receipt_path.exists(), "C85U V2 authorization already consumed")
    attempt_id = uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    staging = output / f".final_acceptance_bundle.staging-{attempt_id}"
    lifecycle_path = staging / "C85U_LIFECYCLE.jsonl"
    lifecycle = _create_lifecycle(
        lifecycle_path, binding_sha=binding_sha, lock_sha=lock_sha,
        attempt_id=attempt_id, output_root=output,
    )
    provisional = C85UExecutionContextV2(
        authorization_file_sha256=authorization_sha,
        authorization_binding_sha256=binding_sha,
        authorization_id=str(record["authorization_id"]),
        execution_lock_path=Path(execution_lock).resolve(),
        execution_lock_sha256=lock_sha,
        execution_lock_commit=lock_commit,
        attempt_id=attempt_id,
        output_root=output,
        receipt_path=receipt_path,
        receipt_sha256="0" * 64,
        head=head,
        lifecycle_path=lifecycle_path,
        acceptance_staging_root=staging,
    )
    lifecycle.append("PREFLIGHT_COMPLETED", context=provisional)
    receipt = {
        "schema_version": CONSUMPTION_SCHEMA_V2,
        "authorized_stage": "C85U",
        **_context_identity(provisional),
        "consumed_at_utc": utc_now(),
    }
    _write_exclusive_fsynced(receipt_path, canonical_json_bytes(receipt))
    context = replace(provisional, receipt_sha256=sha256_file(receipt_path))
    lifecycle.append(
        "AUTHORIZATION_CONSUMED", context=context,
        artifact_or_receipt_sha256=context.receipt_sha256,
    )
    (staging / "authorization_consumed.json").write_bytes(receipt_path.read_bytes())
    validate_execution_context_v2(context)
    return context, lock


def _observed_registry_digest(
    rows: list[dict[str, Any]], *, sidecar: bool,
) -> str:
    prefix = "target_sidecar" if sidecar else "target_artifact"
    return canonical_sha256([
        {
            "unit_id": row["unit_id"],
            "path": row[f"{prefix}_path"],
            "bytes": int(row[f"{prefix}_bytes"]),
            "sha256": row[f"{prefix}_sha256"],
        }
        for row in rows
    ])


def replay_protected_inputs_v2(
    context: C85UExecutionContextV2, registry: U1RuntimeRegistry,
) -> C85UExecutionContextV2:
    """Hash every protected input after authorization and freeze receipt V2."""
    validate_execution_context_v2(context)
    lifecycle = AppendOnlyLifecycleV2(context.lifecycle_path)
    lifecycle.append("PROTECTED_INPUT_REPLAY_STARTED", context=context)
    label_path = registry.evaluation_label_table_path
    require(sha256_file(label_path) == registry.evaluation_label_table_sha256,
            "C85U V2 evaluation label-table SHA drift")
    observed: list[dict[str, Any]] = []
    for row in registry.target_artifact_rows:
        target = Path(str(row["target_artifact_path"]))
        sidecar = Path(str(row["target_sidecar_path"]))
        target_sha = sha256_file(target)
        sidecar_sha = sha256_file(sidecar)
        require(target_sha == row["target_artifact_sha256"],
                f"C85U V2 target-artifact SHA drift: {row['unit_id']}")
        require(sidecar_sha == row["target_sidecar_sha256"],
                f"C85U V2 target-sidecar SHA drift: {row['unit_id']}")
        observed.append({
            "unit_id": row["unit_id"],
            "target_artifact_path": str(target),
            "target_artifact_bytes": target.stat().st_size,
            "target_artifact_sha256": target_sha,
            "target_sidecar_path": str(sidecar),
            "target_sidecar_bytes": sidecar.stat().st_size,
            "target_sidecar_sha256": sidecar_sha,
        })
    require(len(observed) == 1_944, "C85U V2 protected artifact count drift")
    target_bytes = sum(int(row["target_artifact_bytes"]) for row in observed)
    require(target_bytes == registry.target_artifact_total_bytes
            == EXPECTED_TARGET_ARTIFACT_BYTES,
            "C85U V2 protected artifact byte total drift")
    target_registry_sha = _observed_registry_digest(observed, sidecar=False)
    sidecar_registry_sha = _observed_registry_digest(observed, sidecar=True)
    require(target_registry_sha == registry.target_artifact_registry_sha256 and
            sidecar_registry_sha == registry.target_sidecar_registry_sha256,
            "C85U V2 protected artifact registry drift")
    receipt = {
        "schema_version": PROTECTED_REPLAY_SCHEMA_V2,
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "evaluation_label_table_sha256": registry.evaluation_label_table_sha256,
        "evaluation_label_table_rows": registry.evaluation_label_table_rows,
        "evaluation_view_manifest_sha256": registry.evaluation_view_manifest_sha256,
        "target_artifact_rows": len(observed),
        "target_artifact_total_bytes": target_bytes,
        "target_artifact_registry_sha256": target_registry_sha,
        "target_sidecar_rows": len(observed),
        "target_sidecar_registry_sha256": sidecar_registry_sha,
        "replay_completed_at_utc": utc_now(),
        "status": "PASS_PROTECTED_INPUTS_REPLAYED_AFTER_AUTHORIZATION_CONSUMPTION",
    }
    receipt_path = context.output_root / "C85U_PROTECTED_INPUT_REPLAY_V2.json"
    _write_exclusive_fsynced(receipt_path, canonical_json_bytes(receipt))
    updated = replace(
        context, protected_replay_path=receipt_path,
        protected_replay_sha256=sha256_file(receipt_path),
    )
    lifecycle.append(
        "PROTECTED_INPUT_REPLAY_COMPLETED", context=updated,
        artifact_or_receipt_sha256=updated.protected_replay_sha256,
    )
    validate_protected_replay_receipt_v2(updated, registry)
    return updated


def validate_protected_replay_receipt_v2(
    context: C85UExecutionContextV2, registry: U1RuntimeRegistry,
    *, expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    validate_execution_context_v2(context)
    require(context.protected_replay_path is not None and
            context.protected_replay_sha256 is not None,
            "C85U V2 protected replay receipt absent")
    path = context.protected_replay_path
    digest = sha256_file(path)
    require(digest == context.protected_replay_sha256 and
            (expected_receipt_sha256 is None or digest == expected_receipt_sha256),
            "C85U V2 protected replay receipt SHA drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": PROTECTED_REPLAY_SCHEMA_V2,
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "evaluation_label_table_sha256": registry.evaluation_label_table_sha256,
        "evaluation_label_table_rows": registry.evaluation_label_table_rows,
        "evaluation_view_manifest_sha256": registry.evaluation_view_manifest_sha256,
        "target_artifact_rows": len(registry.target_artifact_rows),
        "target_artifact_total_bytes": registry.target_artifact_total_bytes,
        "target_artifact_registry_sha256": registry.target_artifact_registry_sha256,
        "target_sidecar_rows": len(registry.target_artifact_rows),
        "target_sidecar_registry_sha256": registry.target_sidecar_registry_sha256,
        "status": "PASS_PROTECTED_INPUTS_REPLAYED_AFTER_AUTHORIZATION_CONSUMPTION",
    }
    require(all(value.get(key) == expected_value for key, expected_value in expected.items()),
            "C85U V2 protected replay semantic linkage drift")
    require(isinstance(value.get("replay_completed_at_utc"), str),
            "C85U V2 protected replay timestamp absent")
    return value


def stage_receipt_path(context: C85UExecutionContextV2, stage: str) -> Path:
    require(stage in {"U1", "U2"}, "C85U V2 unknown stage receipt")
    return context.output_root / ".stage_receipts" / f"{stage}-{context.attempt_id}.json"


def create_stage_receipt_v2(
    context: C85UExecutionContextV2, stage: str,
    *, prerequisite_sha256: str,
) -> tuple[Path, str]:
    validate_execution_context_v2(context)
    events = AppendOnlyLifecycleV2(context.lifecycle_path).replay()
    expected_last = "STAGE_U1_STARTED" if stage == "U1" else "STAGE_U2_STARTED"
    require(events[-1]["stage"] == expected_last,
            f"C85U V2 {stage} stage lifecycle prerequisite drift")
    if stage == "U1":
        require(context.protected_replay_sha256 == prerequisite_sha256,
                "C85U V2 U1 protected replay prerequisite drift")
    receipt = {
        "schema_version": STAGE_RECEIPT_SCHEMA_V2,
        "stage": stage,
        **_context_identity(context),
        "prerequisite_sha256": prerequisite_sha256,
        "created_at_utc": utc_now(),
    }
    path = stage_receipt_path(context, stage)
    _write_exclusive_fsynced(path, canonical_json_bytes(receipt))
    return path, sha256_file(path)


def validate_stage_receipt_v2(
    context: C85UExecutionContextV2, stage: str, *, prerequisite_sha256: str,
) -> tuple[Path, str]:
    validate_execution_context_v2(context)
    path = stage_receipt_path(context, stage)
    require(path.is_file(), f"C85U V2 {stage} stage receipt absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": STAGE_RECEIPT_SCHEMA_V2,
        "stage": stage,
        **_context_identity(context),
        "prerequisite_sha256": prerequisite_sha256,
    }
    require(all(value.get(key) == item for key, item in expected.items()),
            f"C85U V2 {stage} stage receipt linkage drift")
    return path, sha256_file(path)


def execution_context_record_v2(context: C85UExecutionContextV2) -> dict[str, Any]:
    validate_execution_context_v2(context)
    require(context.protected_replay_path is not None and
            context.protected_replay_sha256 is not None,
            "C85U V2 context serialization precedes protected replay")
    return {
        "schema_version": CONTEXT_SCHEMA_V2,
        **_context_identity(context),
        "receipt_path": str(context.receipt_path),
        "receipt_sha256": context.receipt_sha256,
        "lifecycle_path": str(context.lifecycle_path),
        "acceptance_staging_root": str(context.acceptance_staging_root),
        "protected_replay_path": str(context.protected_replay_path),
        "protected_replay_sha256": context.protected_replay_sha256,
    }


def load_execution_context_record_v2(path: str | Path) -> C85UExecutionContextV2:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value.get("schema_version") == CONTEXT_SCHEMA_V2,
            "C85U V2 execution-context schema drift")
    context = C85UExecutionContextV2(
        authorization_file_sha256=str(value["authorization_file_sha256"]),
        authorization_binding_sha256=str(value["authorization_binding_sha256"]),
        authorization_id=str(value["authorization_id"]),
        execution_lock_path=Path(str(value["execution_lock_path"])).resolve(),
        execution_lock_sha256=str(value["execution_lock_sha256"]),
        execution_lock_commit=str(value["execution_lock_commit"]),
        attempt_id=str(value["attempt_id"]),
        output_root=Path(str(value["output_root"])).resolve(),
        receipt_path=Path(str(value["receipt_path"])).resolve(),
        receipt_sha256=str(value["receipt_sha256"]),
        head=str(value["HEAD"]),
        lifecycle_path=Path(str(value["lifecycle_path"])).resolve(),
        acceptance_staging_root=Path(str(value["acceptance_staging_root"])).resolve(),
        protected_replay_path=Path(str(value["protected_replay_path"])).resolve(),
        protected_replay_sha256=str(value["protected_replay_sha256"]),
    )
    validate_execution_context_v2(context)
    return context


@dataclass
class RuntimeOpenPolicyV2:
    allowed_paths: frozenset[Path]
    allowed_opens: int = 0
    forbidden_opens: int = 0

    def require_allowed(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        if resolved not in self.allowed_paths:
            self.forbidden_opens += 1
            raise RuntimeError(f"C85U V2 runtime file-open policy rejected: {resolved}")
        self.allowed_opens += 1
        return resolved


__all__ = [
    "AUTHORIZATION_PATH_NAME_V2",
    "AUTHORIZATION_SCHEMA_V2",
    "AppendOnlyLifecycleV2",
    "C85UExecutionContextV2",
    "CONTEXT_SCHEMA_V2",
    "DIRECT_STATEMENT",
    "LIFECYCLE_SCHEMA_V2",
    "LOCK_SCHEMA_V2",
    "LOCK_STATUS_V2",
    "PROTECTED_REPLAY_SCHEMA_V2",
    "RuntimeOpenPolicyV2",
    "SUCCESS_LIFECYCLE_V2",
    "authorization_binding_sha256",
    "create_execution_context_v2",
    "create_stage_receipt_v2",
    "execution_context_record_v2",
    "expected_consumption_path",
    "expected_output_root",
    "load_execution_context_record_v2",
    "replay_execution_lock_v2",
    "replay_protected_inputs_v2",
    "stage_receipt_path",
    "utc_now",
    "validate_execution_context_v2",
    "validate_context_lock_v2",
    "validate_protected_replay_receipt_v2",
    "validate_stage_receipt_v2",
]
