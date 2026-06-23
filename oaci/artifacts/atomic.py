"""Atomic artifact-directory commit.

Files are written into a sibling ``.tmp-oaci-artifact-*`` staging dir, each fsync'd, the index written,
``COMMITTED.json`` written LAST, then the staging dir is renamed into place and the parent fsync'd. A
directory without a valid ``COMMITTED.json`` is never a result. No symlink is followed; ``..`` / absolute
/ empty path components are rejected; an existing destination is not overwritten unless asked.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

from .canonical_json import canonical_json_bytes

COMMIT_MARKER = "COMMITTED.json"
INDEX_NAME = "artifact_index.json"


def _reject_symlink_components(final_path: str) -> None:
    """Reject if ANY existing component from the filesystem root down to the final parent is a symlink
    (not just the immediate parent)."""
    parts = os.path.abspath(final_path).split(os.sep)
    cur = os.sep
    for p in parts[1:-1]:                       # every ancestor dir, excluding the (not-yet-existing) leaf
        cur = os.path.join(cur, p)
        if os.path.islink(cur):
            raise ValueError(f"refusing to write under a symlinked path component: {cur}")


def safe_relpath(rel: str) -> str:
    if os.path.isabs(rel):
        raise ValueError(f"artifact path must be relative: {rel!r}")
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe artifact path component in {rel!r}")
    return rel


def _fsync_dir(path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StagingDir:
    """Collects (relpath -> bytes|file-written) into a staging dir; commit() renames it into place."""

    def __init__(self, final_path, *, overwrite=False):
        self.final_path = os.path.abspath(final_path)
        self.parent = os.path.dirname(self.final_path)
        self.overwrite = overwrite
        self._files = []                       # (relpath, abspath)
        self._dirs = set()
        self.staging = None

    def __enter__(self):
        _reject_symlink_components(self.final_path)
        if os.path.islink(self.final_path):
            raise ValueError("refusing to write through a symlink")
        if os.path.exists(self.final_path) and not self.overwrite:
            raise FileExistsError(f"artifact destination already exists: {self.final_path}")
        os.makedirs(self.parent, exist_ok=True)
        # a truly unique staging dir: concurrent writers to the SAME destination never collide and
        # never delete each other's staging.
        self.staging = tempfile.mkdtemp(prefix=".tmp-oaci-artifact-", dir=self.parent)
        return self

    def _abs(self, rel):
        rel = safe_relpath(rel)
        ap = os.path.join(self.staging, rel)
        d = os.path.dirname(ap)
        if d and d not in self._dirs:
            os.makedirs(d, exist_ok=True); self._dirs.add(d)
        return ap

    def write_bytes(self, rel, data: bytes) -> None:
        ap = self._abs(rel)
        with open(ap, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        self._files.append((rel, ap))

    def file_path(self, rel) -> str:
        """Reserve a path for a caller that writes the file itself (e.g. torch.save)."""
        ap = self._abs(rel)
        self._files.append((rel, ap))
        return ap

    def commit(self, index_entries: list, marker_body: dict):
        # index lists every file except itself and the commit marker
        index_doc = {"files": sorted(index_entries, key=lambda e: e["relative_path"])}
        index_bytes = canonical_json_bytes(index_doc)
        self.write_bytes(INDEX_NAME, index_bytes)
        # fsync all touched dirs
        for d in sorted(self._dirs) + [self.staging]:
            _fsync_dir(d)
        marker = {**marker_body, "artifact_index_sha256": hashlib.sha256(index_bytes).hexdigest()}
        self.write_bytes(COMMIT_MARKER, canonical_json_bytes(marker))   # the marker is written LAST
        _fsync_dir(self.staging)
        if os.path.exists(self.final_path) and not self.overwrite:      # re-check just before the rename
            raise FileExistsError(f"artifact destination appeared during write: {self.final_path}")
        os.rename(self.staging, self.final_path)
        _fsync_dir(self.parent)
        self.staging = None
        return self.final_path

    def __exit__(self, exc_type, exc, tb):
        if self.staging is not None and os.path.isdir(self.staging):
            _rmtree(self.staging)          # only OUR staging dir (mkdtemp) -- never a foreign one
        return False


def _rmtree(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    os.rmdir(path)
