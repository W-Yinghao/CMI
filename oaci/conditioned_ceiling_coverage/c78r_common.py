"""C78R exact-authorization guards and content-addressed paths."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from . import c74_cache
from . import c78r_seed3_src_canary as c78r


LOCK_PATH = c78r.REPORT_DIR / "C78R_AUTHORIZED_EXECUTION_LOCK.json"
LOCK_SHA_PATH = c78r.REPORT_DIR / "C78R_AUTHORIZED_EXECUTION_LOCK.sha256"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def load_protocol() -> tuple[dict[str, Any], str, str]:
    expected = c78r.PROTOCOL_SHA_PATH.read_text().strip()
    observed = c78r.sha256_file(c78r.PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError(f"C78R protocol drift: {observed} != {expected}")
    protocol = json.loads(c78r.PROTOCOL_PATH.read_text())
    token = protocol.get("authorization_token_exact")
    if token != c78r.AUTHORIZATION_TOKEN:
        raise RuntimeError("C78R protocol exact token drift")
    return protocol, observed, token


def protocol_commit() -> str:
    commit = git("log", "-1", "--format=%H", "--", str(c78r.PROTOCOL_PATH))
    if not commit:
        raise RuntimeError("C78R protocol is not committed")
    return commit


def create_execution_lock(cli_token: str) -> dict[str, Any]:
    protocol, protocol_sha, expected = load_protocol()
    if not c78r.authorization_matches(cli_token, expected):
        raise PermissionError("C78R authorization lock requires the exact CLI token")
    commit = protocol_commit()
    if git("merge-base", "--is-ancestor", commit, "HEAD") != "":
        raise RuntimeError("C78R protocol commit is not an ancestor of HEAD")
    implementation = protocol["implementation_files"]
    for item in implementation:
        if c78r.sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"C78R implementation drift before authorization lock: {item['path']}")
    lock = {
        "schema_version": "c78r_authorized_execution_lock_v1",
        "protocol_commit": commit,
        "protocol_sha256": protocol_sha,
        "parent_result_commit": c78r.PARENT_RESULT_COMMIT,
        "authorization": {
            "received": True,
            "received_at_utc": c78r.utc_now(),
            "accepted_channel": "exact_CLI_argument_only",
            "token_field": "authorization_token_exact",
            "exact_token_sha256": sha256_text(expected),
        },
        "scope": protocol["scope"],
        "implementation_files": implementation,
        "implementation_identity_sha256": sha256_text(json.dumps(implementation, sort_keys=True, separators=(",", ":"))),
        "EEG_data_access_before_lock": 0,
        "GPU_submission_before_lock": 0,
        "target_outcome_reads_before_lock": 0,
    }
    LOCK_PATH.write_bytes(c78r.canonical_bytes(lock) + b"\n")
    LOCK_SHA_PATH.write_text(c78r.sha256_file(LOCK_PATH) + "\n")
    return lock


def load_execution_lock() -> dict[str, Any]:
    expected = LOCK_SHA_PATH.read_text().strip()
    observed = c78r.sha256_file(LOCK_PATH)
    if expected != observed:
        raise RuntimeError("C78R execution lock drift")
    lock = json.loads(LOCK_PATH.read_text())
    protocol, protocol_sha, token = load_protocol()
    if lock["protocol_sha256"] != protocol_sha:
        raise RuntimeError("C78R lock/protocol mismatch")
    if lock["authorization"]["exact_token_sha256"] != sha256_text(token):
        raise RuntimeError("C78R execution-lock token digest mismatch")
    for item in lock["implementation_files"]:
        if c78r.sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"C78R locked implementation drift: {item['path']}")
    return lock


def require_authorization(cli_token: str | None) -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = load_execution_lock()
    protocol, protocol_sha, expected = load_protocol()
    if not c78r.authorization_matches(cli_token, expected):
        raise PermissionError("C78R requires the exact CLI authorization token")
    if lock["authorization"]["received"] is not True:
        raise PermissionError("C78R execution lock does not record authorization")
    return lock, protocol, protocol_sha


def campaign_root(lock: dict[str, Any]) -> Path:
    return (
        c78r.EXTERNAL_ROOT / "authorized_SRC_canary"
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


def canonical_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["manifest_sha256"] = sha256_text(json.dumps(body, sort_keys=True, separators=(",", ":")))
    return body


def verify_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    supplied = payload.pop("manifest_sha256")
    observed = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if supplied != observed:
        raise RuntimeError(f"C78R manifest self-hash mismatch: {path}")
    payload["manifest_sha256"] = supplied
    return payload


def write_manifest(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c74_cache.atomic_json(path, canonical_manifest(payload))
    return verify_manifest(path)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, c78r.canonical_bytes(payload) + b"\n")
    finally:
        os.close(fd)


def require_field_frozen(lock: dict[str, Any]) -> dict[str, Any]:
    field = verify_manifest(field_frozen_path(lock))
    if field["unit_count"] != c78r.EXPECTED_UNITS or field["SRC_count"] != c78r.EXPECTED_UNITS:
        raise RuntimeError("C78R frozen field is not 80 SRC units")
    return field


def unit_rows() -> list[dict[str, Any]]:
    rows = c78r.read_csv(c78r.TABLE_DIR / "SRC_unit_manifest.csv")
    if len(rows) != c78r.EXPECTED_UNITS or {row["regime"] for row in rows} != {"SRC"}:
        raise RuntimeError("C78R committed unit table drift")
    return rows


def checkpoint_units(lock: dict[str, Any]) -> list[dict[str, Any]]:
    field = require_field_frozen(lock)
    units = sorted(field["units"], key=lambda row: (int(row["level"]), int(row["trajectory_order"])))
    if len(units) != c78r.EXPECTED_UNITS:
        raise RuntimeError("C78R checkpoint unit coverage drift")
    return units


def verify_git_boundary() -> dict[str, Any]:
    lock = load_execution_lock()
    commit = protocol_commit()
    if git("merge-base", "--is-ancestor", commit, "HEAD") != "":
        raise RuntimeError("C78R protocol no longer ancestors execution HEAD")
    dirty = git("status", "--porcelain", "--", *[item["path"] for item in lock["implementation_files"]])
    if dirty:
        raise RuntimeError(f"C78R locked implementation worktree is dirty: {dirty}")
    return {"HEAD": git("rev-parse", "HEAD"), "protocol_commit": commit, "implementation_dirty": False}
