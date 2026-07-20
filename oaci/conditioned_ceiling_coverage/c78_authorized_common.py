"""Shared guards and immutable paths for the authorized C78 P1 campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from . import c74_cache
from . import c78_seed3_instrumented_pilot as c78


LOCK_PATH = c78.REPORT_DIR / "C78_AUTHORIZED_P1_EXECUTION_LOCK.json"
LOCK_SHA_PATH = c78.REPORT_DIR / "C78_AUTHORIZED_P1_EXECUTION_LOCK.sha256"
NO_AUTH_RESULT_COMMIT = "67bca01949c88ab58360179d19868aa78cfc93b7"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def load_execution_lock() -> dict[str, Any]:
    expected = LOCK_SHA_PATH.read_text().strip()
    observed = c78.sha256_file(LOCK_PATH)
    if observed != expected:
        raise RuntimeError(f"C78 authorized execution lock drift: {observed} != {expected}")
    lock = json.loads(LOCK_PATH.read_text())
    protocol, protocol_sha, token, field = c78.load_protocol()
    if lock["protocol_sha256"] != protocol_sha:
        raise RuntimeError("C78 authorized lock protocol hash mismatch")
    if lock["authorization"]["token_field_path"] != field:
        raise RuntimeError("C78 authorized lock token field mismatch")
    if lock["authorization"]["exact_token_sha256"] != sha256_text(token):
        raise RuntimeError("C78 authorized lock token digest mismatch")
    for item in lock["implementation_files"]:
        path = Path(item["path"])
        if c78.sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"C78 authorized implementation drift: {path}")
    return lock


def require_authorization(cli_token: str | None) -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = load_execution_lock()
    protocol, protocol_sha, exact_token, _ = c78.load_protocol()
    if not c78.authorization_matches(cli_token, exact_token):
        raise PermissionError("C78 P1 requires the exact CLI authorization token")
    if lock["authorization"]["received"] is not True:
        raise PermissionError("C78 authorized execution lock does not record PM authorization")
    code_config = c78._code_config_replay(protocol)
    if len(code_config) != 7 or not all(int(row["byte_exact"]) == 1 for row in code_config):
        raise RuntimeError("C78 historical ERM/OACI code or config drift before P1")
    return lock, protocol, protocol_sha


def campaign_root(lock: dict[str, Any]) -> Path:
    return (
        c78.EXTERNAL_ROOT / "authorized_P1"
        / f"protocol_{lock['protocol_sha256'][:16]}"
        / f"implementation_{lock['implementation_identity_sha256'][:16]}"
    )


def field_frozen_path(lock: dict[str, Any]) -> Path:
    return campaign_root(lock) / "gates" / "FIELD_FROZEN.json"


def primary_input_gate_path(lock: dict[str, Any]) -> Path:
    return campaign_root(lock) / "gates" / "PRIMARY_INPUT_VIEWS.json"


def label_view_gate_path(lock: dict[str, Any]) -> Path:
    return campaign_root(lock) / "gates" / "LABEL_VIEWS.json"


def instrumentation_gate_path(lock: dict[str, Any]) -> Path:
    return campaign_root(lock) / "gates" / "INSTRUMENTATION_COMPLETE.json"


def unit_rows() -> list[dict[str, str]]:
    rows = read_csv(c78.TABLE_DIR / "c78_unit_manifest.csv")
    if len(rows) != 82 or len({row["unit_id"] for row in rows}) != 82:
        raise RuntimeError("C78 locked unit manifest is not 82 unique rows")
    if {row["target"] for row in rows} != {"4"} or {row["seed"] for row in rows} != {"3"}:
        raise RuntimeError("C78 unit target/seed drift")
    if {row["level"] for row in rows} != {"0", "1"}:
        raise RuntimeError("C78 unit level drift")
    if {row["regime"] for row in rows} != {"ERM", "OACI"}:
        raise RuntimeError("C78 unit regime drift")
    return rows


def canonical_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["manifest_sha256"] = sha256_text(
        json.dumps(body, sort_keys=True, separators=(",", ":"))
    )
    return body


def verify_canonical_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text())
    supplied = payload.pop("manifest_sha256")
    observed = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if supplied != observed:
        raise RuntimeError(f"C78 manifest self-hash mismatch: {path}")
    payload["manifest_sha256"] = supplied
    return payload


def write_manifest(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = canonical_manifest(payload)
    c74_cache.atomic_json(path, body)
    return verify_canonical_manifest(path)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_git_execution_boundary() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    status = git("status", "--porcelain")
    if branch != "oaci":
        raise RuntimeError(f"C78 P1 requires branch oaci, got {branch}")
    if status:
        raise RuntimeError("C78 P1 requires a clean worktree")
    return {"branch": branch, "commit": git("rev-parse", "HEAD"), "worktree_clean": True}


def checkpoint_sidecars(lock: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = verify_canonical_manifest(field_frozen_path(lock))
    rows = list(frozen["units"])
    if len(rows) != 82 or len({row["unit_id"] for row in rows}) != 82:
        raise RuntimeError("C78 frozen field is not 82 unique units")
    for row in rows:
        for key in ("checkpoint_path", "checkpoint_file_sha256", "checkpoint_id", "sidecar_path", "sidecar_sha256"):
            if not row.get(key):
                raise RuntimeError(f"C78 frozen unit missing {key}: {row.get('unit_id')}")
        if c78.sha256_file(row["checkpoint_path"]) != row["checkpoint_file_sha256"]:
            raise RuntimeError(f"C78 checkpoint file drift: {row['unit_id']}")
        if c78.sha256_file(row["sidecar_path"]) != row["sidecar_sha256"]:
            raise RuntimeError(f"C78 checkpoint sidecar drift: {row['unit_id']}")
    return sorted(rows, key=lambda row: (int(row["level"]), row["regime"], int(row["trajectory_order"])))


def require_field_frozen(lock: dict[str, Any]) -> dict[str, Any]:
    path = field_frozen_path(lock)
    if not path.is_file():
        raise RuntimeError("C78 target/source post-freeze access forbidden before FIELD_FROZEN")
    frozen = verify_canonical_manifest(path)
    if not frozen["all_82_retention_decisions_frozen"]:
        raise RuntimeError("C78 field freeze gate is false")
    if frozen["execution"]["target_data_rows_loaded_during_training"] != 0:
        raise RuntimeError("C78 training target-data isolation failed")
    return frozen
