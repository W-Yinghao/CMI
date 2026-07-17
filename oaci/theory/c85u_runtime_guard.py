"""Single-use authorization and protected-input guard for future C85U execution."""
from __future__ import annotations

from dataclasses import dataclass, replace
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

from oaci.multidataset.c84s_common import require, sha256_file

from .c85u_input_registry import (
    EVALUATION_LABEL_SHA256,
    EVALUATION_LABEL_TABLE,
    FrozenInputRegistry,
    build_frozen_input_registry,
)


AUTHORIZATION_SCHEMA = "c85u_direct_pi_authorization_record_v1"
CONSUMPTION_SCHEMA = "c85u_authorization_consumption_receipt_v1"
LOCK_SCHEMA = "c85u_execution_lock_v1"
LOCK_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
DIRECT_STATEMENT = "授权 C85U"
LOCK_PATH_NAME = "C85U_EXECUTION_LOCK.json"
AUTHORIZATION_PATH_NAME = "C85U_PI_AUTHORIZATION_RECORD.json"
PROTECTED_FALSE_FIELDS = (
    "C85E", "C86", "active_acquisition", "real_data",
    "new_data_or_model_zoo", "manuscript",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONSUMPTION_PATH_MARKER = "C85U_DERIVED_CONSUMPTION_PATH"


def _canonical_json_bytes(value: Any) -> bytes:
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
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def expected_output_root(
    parent: str | Path, lock_sha256: str, authorization_id: str,
) -> Path:
    require(_HEX64.fullmatch(lock_sha256) is not None,
            "C85U invalid execution-lock SHA")
    require(_valid_authorization_id(authorization_id), "C85U invalid authorization ID")
    compact = authorization_id.replace("-", "").lower()[:16]
    return Path(parent).resolve() / f"c85u-{lock_sha256[:16]}-{compact}"


def expected_consumption_path(root: str | Path, binding_sha256: str) -> Path:
    require(_HEX64.fullmatch(binding_sha256) is not None,
            "C85U invalid authorization binding SHA")
    return Path(root).resolve() / f"{binding_sha256}.json"


def replay_execution_lock(
    lock_path: str | Path,
) -> tuple[dict[str, Any], str, Path, str]:
    path = Path(lock_path).resolve()
    sidecar = path.with_suffix(".sha256")
    require(path.name == LOCK_PATH_NAME and path.parent.name == "reports" and
            path.parent.parent.name == "oaci", "C85U lock path drift")
    repo_root = path.parents[2]
    require(path.is_file() and sidecar.is_file(), "C85U lock or sidecar absent")
    lock_sha = sha256_file(path)
    sidecar_tokens = sidecar.read_text(encoding="ascii").split()
    require(sidecar_tokens == [lock_sha, path.name], "C85U lock sidecar drift")
    lock = json.loads(path.read_text(encoding="utf-8"))
    require(lock.get("schema_version") == LOCK_SCHEMA and
            lock.get("status") == LOCK_STATUS and
            lock.get("authorized") is False, "C85U execution-lock state drift")
    require(Path(str(lock["repo_root"])).resolve() == repo_root,
            "C85U execution-lock repository drift")
    bound = lock.get("bound_repository_objects")
    require(isinstance(bound, list) and len(bound) == lock.get("runtime_bound_object_count"),
            "C85U bound-object registry drift")
    for row in bound:
        relative = str(row["path"])
        candidate = repo_root / relative
        require(candidate.is_file() and candidate.stat().st_size == int(row["size_bytes"]),
                f"C85U bound object path/size drift: {relative}")
        require(sha256_file(candidate) == str(row["sha256"]) and
                _git(repo_root, "hash-object", "--", relative) == str(row["git_blob"]),
                f"C85U bound object byte/Git drift: {relative}")
    registry = lock["runtime_bound_registry"]
    registry_path = repo_root / str(registry["path"])
    require(registry_path.is_file() and
            registry_path.stat().st_size == int(registry["size_bytes"]) and
            sha256_file(registry_path) == str(registry["sha256"]),
            "C85U runtime-bound registry drift")
    relative_lock = str(path.relative_to(repo_root))
    lock_commit = _git(repo_root, "log", "-1", "--format=%H", "--", relative_lock)
    require(bool(lock_commit), "C85U lock is not committed")
    return lock, lock_sha, repo_root, lock_commit


def _replay_repository_state(
    repo_root: Path, lock: Mapping[str, Any], lock_commit: str,
) -> str:
    require(_git(repo_root, "branch", "--show-current") == "oaci",
            "C85U execution requires branch oaci")
    require(not _git(repo_root, "status", "--porcelain"),
            "C85U execution requires clean worktree")
    head = _git(repo_root, "rev-parse", "HEAD")
    require(head == _git(repo_root, "rev-parse", "origin/oaci"),
            "C85U execution requires HEAD == origin/oaci")
    require(_is_ancestor(repo_root, str(lock["implementation_commit"]), lock_commit) and
            _is_ancestor(repo_root, lock_commit, head),
            "C85U implementation/lock/HEAD chronology drift")
    return head


def _replay_environment_and_storage(lock: Mapping[str, Any]) -> None:
    environment = lock["environment"]
    require(str(Path(sys.executable).resolve()) == str(environment["python_executable"]),
            "C85U Python executable drift")
    require(sys.version.split()[0] == str(environment["python_version"]),
            "C85U Python version drift")
    require(np.__version__ == str(environment["numpy_version"]) and
            sha256_file(Path(np.__file__)) == str(environment["numpy_file_sha256"]),
            "C85U NumPy identity drift")
    require(scipy.__version__ == str(environment["scipy_version"]) and
            sha256_file(Path(scipy.__file__)) == str(environment["scipy_file_sha256"]),
            "C85U SciPy identity drift")
    output_parent = Path(str(lock["output_root_policy"]["parent"])).resolve()
    existing = output_parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    require(shutil.disk_usage(existing).free >= int(lock["output_root_policy"]["max_bytes"]),
            "C85U output filesystem lacks the locked storage envelope")


def _validate_authorization(
    *,
    authorization_path: Path,
    cli_output_root: Path,
    lock: Mapping[str, Any],
    lock_sha: str,
    lock_commit: str,
    repo_root: Path,
) -> tuple[dict[str, Any], str, str]:
    path = authorization_path.resolve()
    canonical_path = repo_root / "oaci/reports" / AUTHORIZATION_PATH_NAME
    require(path == canonical_path and path.is_file(),
            "C85U authorization must use the committed canonical path")
    relative = str(path.relative_to(repo_root))
    _git(repo_root, "ls-files", "--error-unmatch", relative)
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "direct_explicit_PI_authorization": True,
        "direct_statement_exact": DIRECT_STATEMENT,
        "authorized_stage": "C85U",
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
    }
    for key, expected in required.items():
        require(record.get(key) == expected, f"C85U authorization binding drift: {key}")
    for key in PROTECTED_FALSE_FIELDS:
        require(record.get(key) is False, f"C85U protected authorization field drift: {key}")
    authorization_id = record.get("authorization_id")
    require(_valid_authorization_id(authorization_id), "C85U authorization ID malformed")
    expected_root = expected_output_root(
        lock["output_root_policy"]["parent"], lock_sha, str(authorization_id),
    )
    require(Path(str(record.get("output_root"))).resolve() == expected_root and
            cli_output_root.resolve() == expected_root,
            "C85U authorization output-root binding drift")
    binding_sha = authorization_binding_sha256(record)
    expected_receipt = expected_consumption_path(
        lock["authorization_consumption_root"], binding_sha,
    )
    require(Path(str(record.get("consumption_ledger_path"))).resolve() == expected_receipt,
            "C85U authorization consumption-path binding drift")
    authorization_commit = _git(
        repo_root, "log", "-1", "--format=%H", "--", relative,
    )
    require(_is_ancestor(repo_root, lock_commit, authorization_commit),
            "C85U authorization does not follow the execution lock")
    return record, sha256_file(path), binding_sha


def _write_exclusive_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("C85U authorization was already consumed") from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "C85U authorization receipt short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@dataclass(frozen=True)
class C85UExecutionContext:
    authorization_file_sha256: str
    authorization_binding_sha256: str
    authorization_id: str
    execution_lock_sha256: str
    execution_lock_commit: str
    attempt_id: str
    output_root: Path
    receipt_path: Path
    receipt_sha256: str
    HEAD: str
    protected_replay_path: Path | None = None
    protected_replay_sha256: str | None = None


def validate_execution_context(context: C85UExecutionContext) -> None:
    require(isinstance(context, C85UExecutionContext), "C85U execution context type drift")
    require(context.receipt_path.is_file() and
            sha256_file(context.receipt_path) == context.receipt_sha256,
            "C85U authorization consumption receipt drift")
    receipt = json.loads(context.receipt_path.read_text(encoding="utf-8"))
    required = {
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "HEAD": context.HEAD,
    }
    require(all(receipt.get(key) == value for key, value in required.items()),
            "C85U authorization receipt/context linkage drift")
    if context.protected_replay_path is not None:
        require(context.protected_replay_sha256 is not None and
                context.protected_replay_path.is_file() and
                sha256_file(context.protected_replay_path) == context.protected_replay_sha256,
                "C85U protected-input replay receipt drift")


def create_execution_context(
    *,
    execution_lock: str | Path,
    authorization_record: str | Path,
    output_root: str | Path,
) -> tuple[C85UExecutionContext, FrozenInputRegistry]:
    """Fail closed before protected bytes, then consume one authorization."""
    lock, lock_sha, repo_root, lock_commit = replay_execution_lock(execution_lock)
    head = _replay_repository_state(repo_root, lock, lock_commit)
    _replay_environment_and_storage(lock)
    registry = build_frozen_input_registry()
    output = Path(output_root).resolve()
    record, authorization_sha, binding_sha = _validate_authorization(
        authorization_path=Path(authorization_record), cli_output_root=output,
        lock=lock, lock_sha=lock_sha, lock_commit=lock_commit, repo_root=repo_root,
    )
    receipt_path = expected_consumption_path(
        lock["authorization_consumption_root"], binding_sha,
    )
    require(not output.exists(), "C85U authorization-bound output root already exists")
    require(not receipt_path.exists(), "C85U authorization consumption receipt already exists")
    attempt_id = uuid4().hex
    receipt = {
        "schema_version": CONSUMPTION_SCHEMA,
        "authorized_stage": "C85U",
        "authorization_file_sha256": authorization_sha,
        "authorization_binding_sha256": binding_sha,
        "authorization_id": record["authorization_id"],
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
        "attempt_id": attempt_id,
        "output_root": str(output),
        "HEAD": head,
        "consumed_at_unix_ns": time.time_ns(),
    }
    _write_exclusive_fsynced(receipt_path, _canonical_json_bytes(receipt))
    context = C85UExecutionContext(
        authorization_file_sha256=authorization_sha,
        authorization_binding_sha256=binding_sha,
        authorization_id=str(record["authorization_id"]),
        execution_lock_sha256=lock_sha,
        execution_lock_commit=lock_commit,
        attempt_id=attempt_id,
        output_root=output,
        receipt_path=receipt_path,
        receipt_sha256=sha256_file(receipt_path),
        HEAD=head,
    )
    validate_execution_context(context)
    return context, registry


def replay_protected_inputs_after_consumption(
    context: C85UExecutionContext,
    registry: FrozenInputRegistry,
) -> C85UExecutionContext:
    """Hash labels and target artifacts only after authorization consumption."""
    validate_execution_context(context)
    require(context.output_root.is_dir(), "C85U attempt root absent after consumption")
    require(sha256_file(EVALUATION_LABEL_TABLE) == EVALUATION_LABEL_SHA256,
            "C85U evaluation label-table SHA drift")
    target_bytes = 0
    target_rows: list[dict[str, Any]] = []
    for row in registry.target_artifact_rows:
        path = Path(str(row["target_artifact_path"]))
        observed = sha256_file(path)
        require(observed == str(row["target_artifact_sha256"]),
                f"C85U target artifact SHA drift: {row['unit_id']}")
        target_bytes += path.stat().st_size
        target_rows.append({
            "unit_id": row["unit_id"], "sha256": observed,
            "bytes": path.stat().st_size,
        })
    require(len(target_rows) == 1944, "C85U protected target-artifact coverage drift")
    replay = {
        "schema_version": "c85u_protected_input_replay_receipt_v1",
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "execution_lock_sha256": context.execution_lock_sha256,
        "attempt_id": context.attempt_id,
        "evaluation_label_table_sha256": EVALUATION_LABEL_SHA256,
        "evaluation_label_rows": 4848,
        "target_artifacts": len(target_rows),
        "target_artifact_bytes": target_bytes,
        "target_artifact_registry_sha256": hashlib.sha256(
            _canonical_json_bytes(target_rows)
        ).hexdigest(),
        "status": "PASS_PROTECTED_INPUTS_REPLAYED_AFTER_AUTHORIZATION_CONSUMPTION",
    }
    replay_path = context.output_root / "C85U_PROTECTED_INPUT_REPLAY.json"
    _write_exclusive_fsynced(replay_path, _canonical_json_bytes(replay))
    updated = replace(
        context, protected_replay_path=replay_path,
        protected_replay_sha256=sha256_file(replay_path),
    )
    validate_execution_context(updated)
    return updated


def require_protected_replay(context: C85UExecutionContext) -> None:
    validate_execution_context(context)
    require(context.protected_replay_path is not None,
            "C85U protected input access precedes protected replay")


def execution_context_record(context: C85UExecutionContext) -> dict[str, Any]:
    require_protected_replay(context)
    return {
        "schema_version": "c85u_execution_context_receipt_v1",
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "receipt_path": str(context.receipt_path),
        "receipt_sha256": context.receipt_sha256,
        "HEAD": context.HEAD,
        "protected_replay_path": str(context.protected_replay_path),
        "protected_replay_sha256": context.protected_replay_sha256,
    }


def load_execution_context_record(path: str | Path) -> C85UExecutionContext:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value.get("schema_version") == "c85u_execution_context_receipt_v1",
            "C85U execution-context receipt schema drift")
    context = C85UExecutionContext(
        authorization_file_sha256=str(value["authorization_file_sha256"]),
        authorization_binding_sha256=str(value["authorization_binding_sha256"]),
        authorization_id=str(value["authorization_id"]),
        execution_lock_sha256=str(value["execution_lock_sha256"]),
        execution_lock_commit=str(value["execution_lock_commit"]),
        attempt_id=str(value["attempt_id"]),
        output_root=Path(str(value["output_root"])).resolve(),
        receipt_path=Path(str(value["receipt_path"])).resolve(),
        receipt_sha256=str(value["receipt_sha256"]),
        HEAD=str(value["HEAD"]),
        protected_replay_path=Path(str(value["protected_replay_path"])).resolve(),
        protected_replay_sha256=str(value["protected_replay_sha256"]),
    )
    require_protected_replay(context)
    return context


__all__ = [
    "AUTHORIZATION_SCHEMA", "C85UExecutionContext", "DIRECT_STATEMENT",
    "LOCK_SCHEMA", "LOCK_STATUS", "authorization_binding_sha256",
    "create_execution_context", "expected_consumption_path",
    "execution_context_record", "expected_output_root", "load_execution_context_record",
    "replay_execution_lock",
    "replay_protected_inputs_after_consumption", "require_protected_replay",
    "validate_execution_context",
]
