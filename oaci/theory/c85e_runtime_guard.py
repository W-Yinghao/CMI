"""Receipt-validated C85E runtime context and frozen-file open policy."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "oaci/reports/C85E_EXECUTION_LOCK.json"
AUTHORIZATION_PATH = REPO_ROOT / "oaci/reports/C85E_PI_AUTHORIZATION_RECORD.json"
REGISTRY_PATH = REPO_ROOT / "oaci/reports/c85ep2_tables/c85e_frozen_input_registry.csv"
OUTPUT_PARENT = Path("/projects/EEG-foundation-model/yinghao/oaci-c85e-frozen-field-v1")
CONSUMPTION_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c85e-authorization-consumption-v1"
)
AUTHORIZATION_SCHEMA = "c85e_direct_pi_authorization_record_v1"
LOCK_SCHEMA = "c85e_execution_lock_v1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PATH_TOKENS = (
    "target_evaluation_label_view", "target_construction_label_view",
    "target_logits", "source_arrays", "eeg_arrays", "checkpoints",
)


class C85ERuntimeGuardError(RuntimeError):
    """Raised before any unbound or forbidden C85E input access."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C85ERuntimeGuardError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def authorization_binding_sha256(record: Mapping[str, Any]) -> str:
    value = dict(record)
    value["consumption_ledger_path"] = "<NORMALIZED>"
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def expected_output_root(lock_sha256: str, authorization_id: str) -> Path:
    _require(SHA_RE.fullmatch(lock_sha256) is not None, "invalid C85E lock SHA")
    normalized = UUID(str(authorization_id)).hex
    return OUTPUT_PARENT / f"c85e-v1-{lock_sha256[:16]}-{normalized[:16]}"


def expected_consumption_path(binding_sha256: str) -> Path:
    _require(SHA_RE.fullmatch(binding_sha256) is not None, "invalid authorization binding SHA")
    return CONSUMPTION_ROOT / f"{binding_sha256}.json"


@dataclass(frozen=True)
class FrozenInput:
    object_id: str
    path: Path
    size_bytes: int
    sha256: str
    semantic_role: str


@dataclass(frozen=True)
class ValidatedC85EExecutionContext:
    authorization_file_sha256: str
    authorization_binding_sha256: str
    authorization_id: str
    execution_lock_sha256: str
    execution_lock_commit: str
    attempt_id: str
    output_root: Path
    consumption_receipt_path: Path
    consumption_receipt_sha256: str
    head: str
    inputs: tuple[FrozenInput, ...]


def load_frozen_input_registry(path: str | Path = REGISTRY_PATH) -> tuple[FrozenInput, ...]:
    import csv

    registry = Path(path).resolve()
    _require(registry.is_file(), "C85E frozen-input registry absent")
    rows: list[FrozenInput] = []
    seen: set[str] = set()
    with registry.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == (
            "object_id", "path", "size_bytes", "sha256", "semantic_role", "runtime_access",
        ), "C85E frozen-input registry schema drift")
        for row in reader:
            _require(row["runtime_access"] == "READ_ONLY", "C85E registry access is not read-only")
            target = Path(row["path"]).resolve()
            lowered = str(target).lower()
            _require(not any(token in lowered for token in FORBIDDEN_PATH_TOKENS),
                     f"forbidden C85E runtime path: {target}")
            _require(row["object_id"] not in seen and SHA_RE.fullmatch(row["sha256"]) is not None,
                     "C85E registry identity drift")
            seen.add(row["object_id"])
            rows.append(FrozenInput(
                object_id=row["object_id"], path=target,
                size_bytes=int(row["size_bytes"]), sha256=row["sha256"],
                semantic_role=row["semantic_role"],
            ))
    _require(bool(rows), "C85E frozen-input registry is empty")
    return tuple(rows)


def replay_frozen_inputs(inputs: Sequence[FrozenInput]) -> dict[str, int]:
    files = 0
    total_bytes = 0
    for item in inputs:
        _require(item.path.is_file(), f"C85E frozen input absent: {item.object_id}")
        _require(item.path.stat().st_size == item.size_bytes and
                 sha256_file(item.path) == item.sha256,
                 f"C85E frozen input identity drift: {item.object_id}")
        files += 1
        total_bytes += item.size_bytes
    return {"files": files, "bytes": total_bytes}


def require_registered_path(
    context: ValidatedC85EExecutionContext, path: str | Path,
) -> FrozenInput:
    target = Path(path).resolve()
    matches = [item for item in context.inputs if item.path == target]
    _require(len(matches) == 1, f"C85E attempted unregistered path: {target}")
    item = matches[0]
    _require(target.is_file() and target.stat().st_size == item.size_bytes and
             sha256_file(target) == item.sha256,
             f"C85E registered path changed: {item.object_id}")
    return item


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def _is_ancestor(older: str, newer: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        cwd=REPO_ROOT, capture_output=True,
    ).returncode == 0


def _load_lock(lock_path: Path) -> tuple[dict[str, Any], str, str]:
    _require(lock_path.resolve() == LOCK_PATH.resolve() and lock_path.is_file(),
             "C85E requires the committed operative lock path")
    sidecar = lock_path.with_suffix(".sha256")
    lock_sha = sha256_file(lock_path)
    _require(sidecar.is_file() and sidecar.read_text(encoding="ascii").split()
             == [lock_sha, lock_path.name],
             "C85E lock sidecar drift")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(lock.get("schema_version") == LOCK_SCHEMA and
             lock.get("status") == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED",
             "C85E operative lock state drift")
    bound = lock.get("bound_repository_objects")
    _require(isinstance(bound, list) and len(bound) == lock.get("runtime_bound_object_count"),
             "C85E bound repository-object registry drift")
    seen: set[str] = set()
    for row in bound:
        relative = str(row["path"])
        target = REPO_ROOT / relative
        _require(relative not in seen and target.is_file() and
                 target.stat().st_size == int(row["size_bytes"]) and
                 sha256_file(target) == str(row["sha256"]) and
                 _git("hash-object", "--", relative) == str(row["git_blob"]),
                 f"C85E bound repository object drift: {relative}")
        seen.add(relative)
    registry = lock.get("runtime_bound_registry")
    _require(isinstance(registry, Mapping), "C85E runtime-bound registry absent")
    registry_path = REPO_ROOT / str(registry["path"])
    _require(registry_path.is_file() and
             registry_path.stat().st_size == int(registry["size_bytes"]) and
             sha256_file(registry_path) == str(registry["sha256"]) and
             _git("hash-object", "--", str(registry["path"])) == str(registry["git_blob"]),
             "C85E runtime-bound registry drift")
    lock_commit = _git(
        "log", "-1", "--format=%H", "--", str(lock_path.relative_to(REPO_ROOT)),
    )
    _require(bool(lock_commit), "C85E operative lock is not committed")
    return lock, lock_sha, lock_commit


def _replay_environment(lock: Mapping[str, Any]) -> None:
    import numpy as np

    environment = lock["environment"]
    _require(str(Path(sys.executable).resolve()) == str(environment["python_executable"]) and
             sys.version.split()[0] == str(environment["python_version"]),
             "C85E Python environment drift")
    _require(np.__version__ == str(environment["numpy_version"]) and
             sha256_file(Path(np.__file__)) == str(environment["numpy_file_sha256"]),
             "C85E NumPy environment drift")


def _validate_authorization(
    path: Path, *, lock: Mapping[str, Any], lock_sha: str,
    lock_commit: str, output_root: Path,
) -> tuple[dict[str, Any], str, str]:
    _require(path.resolve() == AUTHORIZATION_PATH.resolve() and path.is_file(),
             "fresh committed C85E authorization record absent")
    record = json.loads(path.read_text(encoding="utf-8"))
    authorization_sha = sha256_file(path)
    binding_sha = authorization_binding_sha256(record)
    _require(record.get("schema_version") == AUTHORIZATION_SCHEMA and
             record.get("direct_explicit_PI_authorization") is True and
             record.get("direct_statement_exact") == "授权 C85E" and
             record.get("authorized_stage") == "C85E",
             "C85E direct authorization statement drift")
    authorization_id = str(record.get("authorization_id", ""))
    UUID(authorization_id)
    _require(record.get("execution_lock_sha256") == lock_sha and
             record.get("execution_lock_commit") == lock_commit,
             "C85E authorization lock binding drift")
    expected_root = expected_output_root(lock_sha, authorization_id)
    _require(output_root.resolve() == expected_root.resolve() and
             record.get("output_root") == str(expected_root),
             "C85E authorization output-root binding drift")
    expected_receipt = expected_consumption_path(binding_sha)
    _require(record.get("consumption_ledger_path") == str(expected_receipt),
             "C85E authorization consumption path drift")
    protected = (
        "C86", "active_acquisition", "real_data", "new_data_or_model_zoo", "manuscript",
    )
    _require(all(record.get(field) is False for field in protected),
             "C85E authorization protected field drift")
    auth_commit = _git("log", "-1", "--format=%H", "--", str(path.relative_to(REPO_ROOT)))
    _require(bool(auth_commit) and _is_ancestor(lock_commit, auth_commit),
             "C85E authorization chronology drift")
    return record, authorization_sha, binding_sha


def create_validated_execution_context(
    *, execution_lock: str | Path, authorization_record: str | Path,
    output_root: str | Path,
) -> ValidatedC85EExecutionContext:
    """Replay, consume once by O_EXCL, and return the sole registered context."""
    lock, lock_sha, lock_commit = _load_lock(Path(execution_lock))
    head = _git("rev-parse", "HEAD")
    _require(_git("rev-parse", "--abbrev-ref", "HEAD") == "oaci" and
             not _git("status", "--porcelain") and
             head == _git("rev-parse", "origin/oaci"),
             "C85E requires clean synchronized branch oaci")
    _require(_is_ancestor(str(lock["implementation_commit"]), lock_commit) and
             _is_ancestor(lock_commit, head),
             "C85E implementation/lock/HEAD chronology drift")
    _replay_environment(lock)
    final = Path(output_root).resolve()
    _require(not final.exists(), "C85E output root already exists")
    record, auth_sha, binding_sha = _validate_authorization(
        Path(authorization_record), lock=lock, lock_sha=lock_sha,
        lock_commit=lock_commit, output_root=final,
    )
    registry_binding = lock["frozen_input_registry"]
    _require(REGISTRY_PATH.stat().st_size == int(registry_binding["size_bytes"]) and
             sha256_file(REGISTRY_PATH) == str(registry_binding["sha256"]),
             "C85E frozen-input registry byte identity drift")
    inputs = load_frozen_input_registry()
    replay = replay_frozen_inputs(inputs)
    _require(replay["files"] == int(registry_binding["rows"]) and
             replay["bytes"] == int(registry_binding["registered_input_bytes"]),
             "C85E frozen-input coverage differs from lock")
    attempt_id = uuid4().hex
    receipt_path = expected_consumption_path(binding_sha)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "c85e_authorization_consumption_receipt_v1",
        "authorization_file_sha256": auth_sha,
        "authorization_binding_sha256": binding_sha,
        "authorization_id": record["authorization_id"],
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
        "attempt_id": attempt_id,
        "output_root": str(final),
        "HEAD": head,
        "consumed_at_unix_ns": time.time_ns(),
    }
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, canonical_json_bytes(receipt))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(receipt_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return ValidatedC85EExecutionContext(
        authorization_file_sha256=auth_sha,
        authorization_binding_sha256=binding_sha,
        authorization_id=str(record["authorization_id"]),
        execution_lock_sha256=lock_sha,
        execution_lock_commit=lock_commit,
        attempt_id=attempt_id,
        output_root=final,
        consumption_receipt_path=receipt_path,
        consumption_receipt_sha256=sha256_file(receipt_path),
        head=head,
        inputs=inputs,
    )


def revalidate_execution_context(context: ValidatedC85EExecutionContext) -> None:
    _require(type(context) is ValidatedC85EExecutionContext,
             "C85E execution context type drift")
    receipt = json.loads(context.consumption_receipt_path.read_text(encoding="utf-8"))
    _require(sha256_file(context.consumption_receipt_path) == context.consumption_receipt_sha256 and
             receipt.get("attempt_id") == context.attempt_id and
             receipt.get("authorization_binding_sha256") == context.authorization_binding_sha256 and
             receipt.get("execution_lock_sha256") == context.execution_lock_sha256 and
             receipt.get("output_root") == str(context.output_root),
             "C85E consumed execution context drift")


__all__ = [
    "AUTHORIZATION_PATH", "CONSUMPTION_ROOT", "C85ERuntimeGuardError",
    "FrozenInput", "LOCK_PATH", "OUTPUT_PARENT", "ValidatedC85EExecutionContext",
    "authorization_binding_sha256", "create_validated_execution_context",
    "expected_consumption_path", "expected_output_root", "load_frozen_input_registry",
    "replay_frozen_inputs", "require_registered_path", "revalidate_execution_context",
    "sha256_file",
]
