"""Shared metadata-only helpers for the C84S analysis implementation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping


class C84SContractError(RuntimeError):
    """Raised when a locked C84S contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    return sha256_file(target)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    if not values:
        raise C84SContractError(f"refusing empty C84S table: {path}")
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise C84SContractError(f"C84S table field-set drift: {path}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n", extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(values)
    return sha256_file(target)


def atomic_publish_directory(final_root: str | Path, writer: Any) -> Path:
    """Populate a sibling staging directory and atomically publish it."""
    final = Path(final_root)
    if final.exists():
        raise C84SContractError(f"C84S final root already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    try:
        writer(staging)
        os.replace(staging, final)
    except BaseException as primary_error:
        cleanup_notes: list[str] = []
        cleanup_callback = getattr(writer, "cleanup_on_failure", None)
        if cleanup_callback is not None:
            try:
                cleanup_callback()
            except BaseException as cleanup_error:
                cleanup_notes.append(
                    f"writer cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        if staging.exists():
            import shutil
            last_cleanup_error: BaseException | None = None
            for attempt in range(4):
                try:
                    shutil.rmtree(staging)
                    last_cleanup_error = None
                    break
                except BaseException as cleanup_error:
                    last_cleanup_error = cleanup_error
                    if attempt < 3:
                        time.sleep(0.05 * (attempt + 1))
            if last_cleanup_error is not None:
                cleanup_notes.append(
                    "staging cleanup failed after four attempts; residual preserved at "
                    f"{staging}: {type(last_cleanup_error).__name__}: {last_cleanup_error}"
                )
        for note in cleanup_notes:
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(note)
                continue
            try:
                notes = list(getattr(primary_error, "__notes__", ()))
                notes.append(note)
                setattr(primary_error, "__notes__", notes)
            except BaseException:
                # Cleanup diagnostics must never replace the primary failure.
                pass
        raise
    return final


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C84SContractError(message)


def digest_low64(key: str) -> int:
    """Return the locked big-endian integer from the final eight SHA-256 bytes."""
    return int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[-8:], "big")
