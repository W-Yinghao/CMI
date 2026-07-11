"""Fail-closed persistence primitives for STAR training attempts."""

import hashlib
import json
import math
import os
import re
import shutil
import stat
import uuid
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, TextIO, Tuple


ATTEMPT_PATTERN = re.compile(r"attempt_[0-9]{2,}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_temp(temp_path: Path, destination: Path) -> None:
    """Publish by hard link so an existing destination is never replaced."""
    try:
        os.link(str(temp_path), str(destination))
        _fsync_directory(destination.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    if not destination.is_file():
        raise RuntimeError(f"atomic publish failed: {destination}")


def atomic_write_bytes_no_overwrite(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    temporary = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_publish_temp(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json_no_overwrite(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_bytes_no_overwrite(Path(path), canonical_json_bytes(payload))


def atomic_copy_file_no_overwrite(source: Path, destination: Path) -> Dict[str, object]:
    source = Path(source)
    destination = Path(destination)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"copy source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite copied artifact: {destination}")
    source_sha_before = sha256_file(source)
    temporary = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        _atomic_publish_temp(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    source_sha_after = sha256_file(source)
    destination_sha = sha256_file(destination)
    if source_sha_before != source_sha_after or destination_sha != source_sha_before:
        raise RuntimeError("atomic copy SHA stability check failed")
    return {
        "source_sha_before": source_sha_before,
        "source_sha_after": source_sha_after,
        "destination_sha": destination_sha,
    }


def atomic_torch_save_no_overwrite(path: Path, payload: Mapping[str, object]) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    temporary = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_publish_temp(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def claim_attempt_directory(path: Path, attempt_id: str) -> Path:
    """Atomically claim an absent or empty attempt directory once."""
    if not ATTEMPT_PATTERN.fullmatch(str(attempt_id)):
        raise ValueError("attempt_id must match attempt_NN")
    destination = Path(path)
    if destination.name != attempt_id:
        raise ValueError("runtime output directory basename must equal attempt_id")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise FileExistsError(f"attempt output path is not a fresh directory: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(f"attempt output directory is nonempty: {destination}")
    else:
        destination.mkdir()
    lock = destination / ".attempt_lock.json"
    atomic_write_json_no_overwrite(lock, {
        "attempt_id": attempt_id,
        "claim_pid": os.getpid(),
        "overwrite_allowed": False,
    })
    return destination


def open_telemetry(path: Path) -> Tuple[TextIO, "hashlib._Hash"]:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"telemetry already exists: {destination}")
    handle = destination.open("x", encoding="utf-8", buffering=1)
    return handle, hashlib.sha256()


def append_telemetry_row(
    handle: TextIO,
    digest,
    row: Mapping[str, object],
) -> str:
    encoded = canonical_json_bytes(row)
    handle.write(encoded.decode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())
    digest.update(encoded)
    return digest.hexdigest()


def validate_telemetry_file(path: Path, expected_steps: int) -> Dict[str, object]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise RuntimeError(f"invalid telemetry line {line_number}")
            row = json.loads(line)
            if int(row.get("step", -1)) != line_number:
                raise RuntimeError(f"telemetry step sequence breaks at line {line_number}")
            numeric_fields = (
                "loss",
                "encoder_grad_norm_before_clipping",
                "encoder_grad_norm_after_clipping",
                "model_grad_norm_before_clipping",
                "model_grad_norm_after_clipping",
                "temporary_head_grad_norm_before_clipping",
                "temporary_head_grad_norm_after_clipping",
                "parameter_delta_norm",
                "learning_rate",
            )
            if not all(math.isfinite(float(row[field])) for field in numeric_fields):
                raise RuntimeError(f"non-finite telemetry value at line {line_number}")
            if row.get("nan_inf_status") != "PASS":
                raise RuntimeError(f"telemetry finite gate fails at line {line_number}")
            rows.append(row)
    if len(rows) != int(expected_steps):
        raise RuntimeError(f"telemetry rows {len(rows)} != {expected_steps}")
    return {
        "rows": len(rows),
        "first_step": int(rows[0]["step"]),
        "final_step": int(rows[-1]["step"]),
        "sha256": sha256_file(path),
        "all_finite": True,
    }


def no_temporary_files(root: Path) -> bool:
    return not any(".tmp-" in path.name for path in Path(root).rglob("*"))


def freeze_completed_attempt(root: Path) -> None:
    """Remove all write bits after completion.json has been published."""
    root = Path(root)
    completion = root / "completion.json"
    if not completion.is_file():
        raise RuntimeError("attempt cannot be frozen before completion.json")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"completed attempt contains symlink: {path}")
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o555)
    root.chmod(0o555)


def tree_is_read_only(root: Path) -> bool:
    paths: Iterable[Path] = [Path(root), *Path(root).rglob("*")]
    return all(
        path.is_symlink() is False
        and (stat.S_IMODE(path.stat().st_mode) & 0o222) == 0
        for path in paths
    )
