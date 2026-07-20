"""C74 protocol guards and deterministic external-cache primitives.

This module contains no model or EEG loader.  Keeping cache mechanics separate
from inference makes the authorization boundary and the T2/T3-HO role checks
independently testable.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from . import c69_powered_trial_cache_scaleup as c69


AUTH_TOKEN = "C74_T2_SOURCE_WZ_REINFERENCE_AUTHORIZED"
PROTOCOL_PATH = "oaci/reports/C74_T2_SOURCE_WZ_INSTRUMENTATION_PROTOCOL.json"
PROTOCOL_SHA_PATH = "oaci/reports/C74_T2_SOURCE_WZ_INSTRUMENTATION_PROTOCOL.sha256"
T2_MANIFEST_PATH = "oaci/reports/c74_tables/full_t2_unit_manifest.csv"
T3_HO_MANIFEST_PATH = "oaci/reports/c74_tables/t3_ho_holdout_unit_manifest.csv"
C65_MAP_PATH = "oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: str | Path, payload: dict) -> str:
    """Write canonical JSON atomically and return its content hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return digest


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def deterministic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> str:
    """Write an NPZ with fixed ZIP metadata so equal arrays hash identically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_name, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name in sorted(arrays):
                if not name or "/" in name or name.endswith(".npy"):
                    raise ValueError(f"invalid deterministic NPZ field {name!r}")
                array = np.asarray(arrays[name])
                if array.dtype.hasobject:
                    raise TypeError(f"object arrays are forbidden in C74 cache: {name}")
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(array))
        digest = sha256_file(tmp_name)
        if path.exists():
            if sha256_file(path) != digest:
                raise RuntimeError(f"immutable C74 shard collision: {path}")
            os.unlink(tmp_name)
        else:
            os.replace(tmp_name, path)
        return digest
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_content_addressed_npz(
    directory: str | Path,
    kind: str,
    arrays: dict[str, np.ndarray],
) -> dict:
    """Materialize one immutable deterministic shard and return its descriptor."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    provisional = directory / f".{kind}.provisional.npz"
    # A provisional file is never an immutable artifact.  Removing a stale one
    # makes an interrupted unit safely restartable without weakening final-hash
    # collision checks.
    provisional.unlink(missing_ok=True)
    digest = deterministic_npz(provisional, arrays)
    final_path = directory / f"{kind}_sha256_{digest[:16]}.npz"
    if final_path.exists():
        if sha256_file(final_path) != digest:
            raise RuntimeError(f"C74 content-address collision: {final_path}")
        provisional.unlink(missing_ok=True)
    else:
        os.replace(provisional, final_path)
    return {
        "kind": kind,
        "path": str(final_path),
        "sha256": digest,
        "size_bytes": final_path.stat().st_size,
        "fields": sorted(arrays),
        "row_count": int(next(iter(arrays.values())).shape[0]) if arrays else 0,
    }


def verify_shard(descriptor: dict, *, required_fields: set[str] | None = None) -> None:
    path = Path(descriptor["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != descriptor["sha256"]:
        raise RuntimeError(f"C74 shard hash mismatch: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise RuntimeError(f"C74 shard size mismatch: {path}")
    with np.load(path, allow_pickle=False) as shard:
        fields = set(shard.files)
        if required_fields is not None and fields != required_fields:
            raise RuntimeError(f"C74 shard schema mismatch for {path}: {sorted(fields)}")
        lengths = {int(shard[name].shape[0]) for name in shard.files if shard[name].ndim >= 1}
    if len(lengths) > 1:
        raise RuntimeError(f"C74 shard row-count mismatch: {path} -> {sorted(lengths)}")
    observed_rows = next(iter(lengths), 0)
    if observed_rows != int(descriptor["row_count"]):
        raise RuntimeError(f"C74 descriptor row-count mismatch: {path}")


def load_locked_protocol() -> dict:
    expected = Path(PROTOCOL_SHA_PATH).read_text().strip()
    observed = sha256_file(PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError(f"C74 protocol hash drift: expected {expected}, observed {observed}")
    protocol = json.loads(Path(PROTOCOL_PATH).read_text())
    if protocol["authorization"]["exact_token_sha256"] != sha256_text(AUTH_TOKEN):
        raise RuntimeError("C74 protocol authorization digest mismatch")
    for item in protocol["locked_unit_tables"].values():
        path = item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"C74 locked table drift: {path}")
        if Path(path).stat().st_size != int(item["size_bytes"]):
            raise RuntimeError(f"C74 locked table size drift: {path}")
    return protocol


def authorization_ok(token: str) -> bool:
    """Authorization is exact CLI value equality; callers must not inspect env."""
    return token == AUTH_TOKEN


def locked_unit_sets() -> tuple[set[str], set[str]]:
    t2 = {row["checkpoint_id"] for row in read_csv(T2_MANIFEST_PATH)}
    t3 = {row["checkpoint_id"] for row in read_csv(T3_HO_MANIFEST_PATH)}
    if len(t2) != 216 or len(t3) != 1052 or t2 & t3:
        raise RuntimeError("C74 locked T2/T3-HO unit-role invariant failed")
    return t2, t3


def stage_rows(stage: str, target_id: int) -> list[dict]:
    if stage not in {"P0_pilot", "P1_expansion"}:
        raise ValueError(f"invalid C74 stage: {stage}")
    locked = [
        row for row in read_csv(T2_MANIFEST_PATH)
        if row["instrumentation_stage"] == stage and int(row["target_id"]) == int(target_id)
    ]
    expected = 6 if stage == "P0_pilot" else 18
    if len(locked) != expected:
        raise RuntimeError(f"C74 {stage} target {target_id} expected {expected} units, got {len(locked)}")

    mapping_rows = read_csv(C65_MAP_PATH)
    _, _, t2_schedule = c69.build_schedule(mapping_rows)
    by_checkpoint = {row["checkpoint_id"]: row for row in t2_schedule}
    t2_ids, t3_ids = locked_unit_sets()
    resolved = []
    for lock in sorted(locked, key=lambda row: int(row["execution_ordinal"])):
        checkpoint_id = lock["checkpoint_id"]
        if checkpoint_id not in t2_ids or checkpoint_id in t3_ids:
            raise RuntimeError(f"C74 holdout contamination: {checkpoint_id}")
        source = dict(by_checkpoint[checkpoint_id])
        source.update({f"c74_{key}": value for key, value in lock.items()})
        resolved.append(source)
    return resolved


def run_root(protocol: dict) -> Path:
    return Path(protocol["cache"]["external_root"]) / f"protocol_{sha256_file(PROTOCOL_PATH)[:16]}"


def unit_directory(protocol: dict, stage: str, target_id: int, unit_id: str) -> Path:
    return run_root(protocol) / stage / f"target-{int(target_id):03d}" / "units" / unit_id


def stage_gate_path(protocol: dict, stage: str) -> Path:
    return run_root(protocol) / "gates" / f"{stage}_GATE.json"


def verify_unit_manifest(path: str | Path, *, rehash_payloads: bool = True) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload["manifest_sha256"] != sha256_text(
        json.dumps({k: v for k, v in payload.items() if k != "manifest_sha256"}, sort_keys=True, separators=(",", ":"))
    ):
        raise RuntimeError(f"C74 unit-manifest self-hash mismatch: {path}")
    if rehash_payloads:
        for shard in payload["shards"]:
            verify_shard(shard)
    return payload


def self_hashed_manifest(payload: dict) -> dict:
    body = dict(payload)
    body["manifest_sha256"] = sha256_text(json.dumps(body, sort_keys=True, separators=(",", ":")))
    return body
