"""ACAR V5 Stage-1B raw-BIDS recording discovery + manifest (pure/stdlib; nothing heavy at import). Discovery is RAW-BIDS-ONLY and
deterministic: it looks ONLY in a subject's `eeg/` and `ses-*/eeg/` directories, ignores everything else, and fail-closes on
derivatives/sourcedata/code locations, symlinked recordings, or no raw recording. It records a hashed manifest so the exact set of
recordings consumed at the real run is auditable. Windowing across recording boundaries is prevented at the reader (each recording
is windowed independently); this module only decides WHICH files are admissible raw recordings.
"""
from __future__ import annotations
import hashlib
import json
import os

EEG_EXTENSIONS = (".edf", ".bdf", ".set", ".vhdr", ".fif")
EXCLUDED_COMPONENTS = ("derivatives", "sourcedata", "code")   # never treat these as raw EEG


class RawManifestError(RuntimeError):
    pass


def _eeg_dirs(subject_dir):
    """The BIDS raw EEG directories for a subject: <sub>/eeg and <sub>/ses-*/eeg (only). A SYMLINKED eeg/ or session directory is
    rejected fail-closed — a symlink can point into derivatives/sourcedata and normpath would not reveal it."""
    dirs = []
    top = os.path.join(subject_dir, "eeg")
    if os.path.islink(top):
        raise RawManifestError(f"symlinked eeg/ directory rejected: {top}")
    if os.path.isdir(top):
        dirs.append(top)
    for entry in sorted(os.listdir(subject_dir)):
        if entry.startswith("ses-"):
            sesdir = os.path.join(subject_dir, entry)
            if os.path.islink(sesdir):
                raise RawManifestError(f"symlinked session directory rejected: {sesdir}")
            se = os.path.join(sesdir, "eeg")
            if os.path.islink(se):
                raise RawManifestError(f"symlinked eeg/ directory rejected: {se}")
            if os.path.isdir(se):
                dirs.append(se)
    return dirs


def discover_raw_recordings(subject_dir):
    """Deterministic, sorted raw EEG recordings under the subject's eeg/ (and ses-*/eeg/) dirs ONLY. Fail-closed."""
    if not subject_dir or not os.path.isdir(subject_dir):
        raise RawManifestError(f"subject dir not found: {subject_dir}")
    dirs = _eeg_dirs(subject_dir)
    if not dirs:
        raise RawManifestError(f"no BIDS eeg/ directory under {subject_dir}")
    found = []
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() not in EEG_EXTENSIONS:
                continue
            p = os.path.join(d, f)
            if os.path.islink(p):
                raise RawManifestError(f"symlinked recording rejected: {p}")
            if not os.path.isfile(p):
                continue
            parts = set(os.path.realpath(p).split(os.sep))    # realpath resolves symlinks → catches derivatives reached via a link
            if parts & set(EXCLUDED_COMPONENTS) or any(x.startswith("acar_") for x in parts):
                raise RawManifestError(f"recording resolves under an excluded (non-raw) location: {p}")
            found.append(p)
    if not found:
        raise RawManifestError(f"no raw EEG recordings ({EEG_EXTENSIONS}) under {subject_dir}/eeg")
    return sorted(found)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(subject_dir):
    """A hashed, deterministic manifest of the subject's raw recordings (path/sha256/size + a manifest_sha256 over them)."""
    files = discover_raw_recordings(subject_dir)
    entries = [{"path": p, "sha256": _sha256_file(p), "n_bytes": os.path.getsize(p)} for p in files]
    man = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"subject_dir": subject_dir, "files": entries, "manifest_sha256": man}
