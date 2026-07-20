"""C78F scope-bound authorization guard and external artifact ABI."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from . import c74_cache
from . import c78f_full_seed3_field as c78f


LOCK_PATH = c78f.REPORT_DIR / "C78F_AUTHORIZED_EXECUTION_LOCK.json"
LOCK_SHA_PATH = c78f.REPORT_DIR / "C78F_AUTHORIZED_EXECUTION_LOCK.sha256"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["manifest_sha256"] = sha256_text(json.dumps(body, sort_keys=True, separators=(",", ":")))
    return body


def write_manifest(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c74_cache.atomic_json(path, canonical_manifest(payload))
    return verify_manifest(path)


def verify_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text())
    supplied = payload.pop("manifest_sha256")
    observed = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if supplied != observed:
        raise RuntimeError(f"C78F manifest self-hash mismatch: {path}")
    payload["manifest_sha256"] = supplied
    return payload


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, c78f.canonical_bytes(payload) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = c78f.PROTOCOL_SHA_PATH.read_text().strip()
    observed = c78f.sha256_file(c78f.PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError(f"C78F protocol drift: {observed} != {expected}")
    protocol = json.loads(c78f.PROTOCOL_PATH.read_text())
    auth = protocol.get("authorization", {})
    if auth.get("mode") != c78f.AUTHORIZATION_MODE:
        raise RuntimeError("C78F authorization mode drift")
    if auth.get("explicit_user_authorization_received") is not True:
        raise PermissionError("C78F direct user authorization is absent")
    if auth.get("evidence_sha256") != c78f.AUTHORIZATION_EVIDENCE_SHA256:
        raise RuntimeError("C78F direct authorization evidence drift")
    c78s_expected = c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip()
    c78s_observed = c78f.sha256_file(c78f.C78S_PROTOCOL_PATH)
    if c78s_expected != c78s_observed or protocol["C78S_analysis_lock"]["sha256"] != c78s_observed:
        raise RuntimeError("C78F/C78S protocol lock mismatch")
    return protocol, observed


def protocol_commit() -> str:
    commit = git("log", "-1", "--format=%H", "--", str(c78f.PROTOCOL_PATH))
    if not commit:
        raise RuntimeError("C78F protocol is not committed")
    return commit


def create_execution_lock() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    commit = protocol_commit()
    if git("merge-base", "--is-ancestor", commit, "HEAD") != "":
        raise RuntimeError("C78F protocol commit is not an ancestor of HEAD")
    for item in protocol["implementation_files"]:
        if c78f.sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"C78F implementation drift before execution lock: {item['path']}")
    scope_identity = sha256_text(json.dumps(protocol["scope"], sort_keys=True, separators=(",", ":")))
    implementation_identity = sha256_text(
        json.dumps(protocol["implementation_files"], sort_keys=True, separators=(",", ":"))
    )
    lock = {
        "schema_version": "c78f_authorized_execution_lock_v1",
        "protocol_commit": commit,
        "protocol_sha256": protocol_sha,
        "C78S_protocol_sha256": protocol["C78S_analysis_lock"]["sha256"],
        "authorization": {
            "received": True,
            "received_at_utc": c78f.utc_now(),
            "mode": c78f.AUTHORIZATION_MODE,
            "evidence_sha256": c78f.AUTHORIZATION_EVIDENCE_SHA256,
            "magic_token_required": False,
            "scope_bound": True,
        },
        "scope": protocol["scope"],
        "scope_identity_sha256": scope_identity,
        "implementation_files": protocol["implementation_files"],
        "implementation_identity_sha256": implementation_identity,
        "remaining_target_EEG_data_access_before_lock": 0,
        "GPU_submission_before_lock": 0,
        "remaining_target_outcome_reads_before_lock": 0,
        "seed4_access_before_lock": 0,
    }
    LOCK_PATH.write_bytes(c78f.canonical_bytes(lock) + b"\n")
    LOCK_SHA_PATH.write_text(c78f.sha256_file(LOCK_PATH) + "\n")
    return lock


def load_execution_lock() -> dict[str, Any]:
    expected = LOCK_SHA_PATH.read_text().strip()
    observed = c78f.sha256_file(LOCK_PATH)
    if expected != observed:
        raise RuntimeError("C78F execution lock drift")
    lock = json.loads(LOCK_PATH.read_text())
    protocol, protocol_sha = load_protocol()
    if lock["protocol_sha256"] != protocol_sha:
        raise RuntimeError("C78F lock/protocol mismatch")
    if lock["authorization"]["received"] is not True:
        raise PermissionError("C78F execution lock lacks direct authorization")
    if lock["authorization"]["mode"] != c78f.AUTHORIZATION_MODE:
        raise PermissionError("C78F execution lock authorization mode mismatch")
    if lock["authorization"]["evidence_sha256"] != c78f.AUTHORIZATION_EVIDENCE_SHA256:
        raise PermissionError("C78F execution lock authorization evidence mismatch")
    lock_commit = git("log", "-1", "--format=%H", "--", str(LOCK_PATH))
    if not lock_commit or git("merge-base", "--is-ancestor", lock_commit, "HEAD") != "":
        raise RuntimeError("C78F execution lock must be committed before execution")
    for item in lock["implementation_files"]:
        if c78f.sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"C78F locked implementation drift: {item['path']}")
    return lock


def require_authorization() -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = load_execution_lock()
    protocol, protocol_sha = load_protocol()
    return lock, protocol, protocol_sha


def campaign_root(lock: dict[str, Any]) -> Path:
    return (
        c78f.EXTERNAL_ROOT
        / f"protocol_{lock['protocol_sha256'][:16]}"
        / f"implementation_{lock['implementation_identity_sha256'][:16]}"
    )


def target_root(lock: dict[str, Any], target: int) -> Path:
    require_target(target)
    return campaign_root(lock) / "targets" / f"target-{int(target):03d}"


def oaci_field_path(lock: dict[str, Any], target: int) -> Path:
    return target_root(lock, target) / "training" / "oaci_erm" / "FIELD_FROZEN.json"


def src_field_path(lock: dict[str, Any], target: int) -> Path:
    return target_root(lock, target) / "training" / "src" / "FIELD_FROZEN.json"


def primary_view_path(lock: dict[str, Any], target: int) -> Path:
    return target_root(lock, target) / "views" / "PRIMARY_INPUT_VIEWS.json"


def label_view_path(lock: dict[str, Any], target: int) -> Path:
    return target_root(lock, target) / "views" / "LABEL_VIEWS.json"


def instrumentation_path(lock: dict[str, Any], target: int) -> Path:
    return target_root(lock, target) / "instrumentation" / "INSTRUMENTATION_COMPLETE.json"


def wave_gate_path(lock: dict[str, Any], wave: str) -> Path:
    if wave not in {"A", "B"}:
        raise ValueError(wave)
    return campaign_root(lock) / "gates" / f"WAVE_{wave}_ENGINEERING_VALID.json"


def full_field_path(lock: dict[str, Any]) -> Path:
    return campaign_root(lock) / "gates" / "FULL_SEED3_FIELD_FROZEN.json"


def require_target(target: int) -> int:
    target = int(target)
    if target not in c78f.TARGETS:
        raise ValueError(f"C78F target {target} is not in the locked remaining field")
    return target


def units_for(target: int, regimes: set[str] | None = None) -> list[dict[str, str]]:
    target = require_target(target)
    rows = [
        row for row in read_csv(c78f.TABLE_DIR / "full_unit_manifest.csv")
        if int(row["target"]) == target and (regimes is None or row["regime"] in regimes)
    ]
    expected = 162 if regimes is None else (82 if regimes == {"ERM", "OACI"} else 80)
    if len(rows) != expected or len({row["unit_id"] for row in rows}) != expected:
        raise RuntimeError(f"C78F target {target} locked unit subset drift: {len(rows)} != {expected}")
    return sorted(rows, key=lambda row: (int(row["level"]), c78f.REGIMES.index(row["regime"]), int(row["trajectory_order"])))


def require_oaci_field(lock: dict[str, Any], target: int) -> dict[str, Any]:
    field = verify_manifest(oaci_field_path(lock, target))
    if field["unit_count"] != 82 or field["target"] != int(target):
        raise RuntimeError("C78F OACI/ERM field coverage drift")
    if field["execution"]["target_label_reads_during_training"] != 0:
        raise RuntimeError("C78F OACI/ERM target isolation failed")
    return field


def require_src_field(lock: dict[str, Any], target: int) -> dict[str, Any]:
    field = verify_manifest(src_field_path(lock, target))
    if field["unit_count"] != 80 or field["target"] != int(target):
        raise RuntimeError("C78F SRC field coverage drift")
    if field["execution"]["target_label_reads_during_training"] != 0:
        raise RuntimeError("C78F SRC target isolation failed")
    return field


def checkpoint_units(lock: dict[str, Any], target: int) -> list[dict[str, Any]]:
    oaci = require_oaci_field(lock, target)
    src = require_src_field(lock, target)
    units = [*oaci["units"], *src["units"]]
    if len(units) != 162 or len({row["unit_id"] for row in units}) != 162:
        raise RuntimeError(f"C78F target {target} field is not 162 unique units")
    for unit in units:
        for key in ("checkpoint_path", "checkpoint_file_sha256", "sidecar_path", "sidecar_sha256"):
            if not unit.get(key):
                raise RuntimeError(f"C78F unit missing {key}: {unit.get('unit_id')}")
        if c78f.sha256_file(unit["checkpoint_path"]) != unit["checkpoint_file_sha256"]:
            raise RuntimeError(f"C78F checkpoint drift: {unit['unit_id']}")
        if c78f.sha256_file(unit["sidecar_path"]) != unit["sidecar_sha256"]:
            raise RuntimeError(f"C78F sidecar drift: {unit['unit_id']}")
    return sorted(units, key=lambda row: (int(row["level"]), c78f.REGIMES.index(row["regime"]), int(row["trajectory_order"])))


def verify_git_boundary() -> dict[str, Any]:
    lock = load_execution_lock()
    dirty = git("status", "--porcelain", "--", *[item["path"] for item in lock["implementation_files"]])
    if dirty:
        raise RuntimeError(f"C78F locked implementation is dirty: {dirty}")
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "protocol_commit": protocol_commit(),
        "implementation_dirty": False,
    }
