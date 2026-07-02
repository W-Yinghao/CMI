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


def _brainvision_sidecars(vhdr_path):
    """Parse DataFile / MarkerFile from a BrainVision .vhdr the SAME way mne (configparser) does — CASE-INSENSITIVE keys and
    whitespace TOLERANT around '=' — so the audited sidecar set matches the bytes mne will actually consume. Sidecars MUST be a bare
    basename in the SAME eeg dir (no path / escape). Fail-closed on a value that isn't a bare filename."""
    out = []
    with open(vhdr_path, encoding="latin-1", errors="replace") as f:
        text = f.read()
    for line in text.splitlines():
        if "=" not in line:
            continue
        raw_key, raw_val = line.split("=", 1)
        key = raw_key.strip().lower()                         # configparser lowercases keys + strips surrounding whitespace
        if key not in ("datafile", "markerfile"):
            continue
        name = raw_val.strip()
        if not name:
            continue
        if name != os.path.basename(name) or name in (".", ".."):
            raise RawManifestError(f"{vhdr_path}: {key} must be a bare filename in the same eeg dir, got {name!r}")
        out.append(os.path.join(os.path.dirname(vhdr_path), name))
    return out


def resolve_sidecars(primary_path):
    """Format sidecars that mne will actually consume: BrainVision .vhdr→(.eeg/.dat via DataFile, .vmrk via MarkerFile); EEGLAB
    .set→.fdt if present. Each must exist, be a non-symlink regular file in the SAME eeg dir. Fail-closed on missing/symlink/escape."""
    ext = os.path.splitext(primary_path)[1].lower()
    d = os.path.dirname(primary_path)
    sidecars = []
    if ext == ".vhdr":
        sidecars = _brainvision_sidecars(primary_path)
        if not sidecars:
            raise RawManifestError(f"{primary_path}: BrainVision header declares no DataFile/MarkerFile")
    elif ext == ".set":
        fdt = os.path.join(d, os.path.splitext(os.path.basename(primary_path))[0] + ".fdt")
        if os.path.isfile(fdt):                               # .fdt is optional (single-file .set is allowed)
            sidecars = [fdt]
    for sc in sidecars:
        if os.path.islink(sc):
            raise RawManifestError(f"symlinked sidecar rejected: {sc}")
        if not os.path.isfile(sc):
            raise RawManifestError(f"declared sidecar missing: {sc}")
        if os.path.dirname(os.path.realpath(sc)) != os.path.realpath(d):
            raise RawManifestError(f"sidecar escapes the recording's eeg dir: {sc}")
    return sidecars


def build_manifest(subject_dir):
    """A hashed, deterministic manifest of the subject's raw recordings AND their format sidecars (BrainVision .eeg/.vmrk, EEGLAB
    .fdt) — so the exact bytes mne consumes are audited, not just the header path. Each entry: path/sha256/n_bytes/role."""
    primaries = discover_raw_recordings(subject_dir)
    entries = []
    for p in primaries:
        entries.append({"path": p, "sha256": _sha256_file(p), "n_bytes": os.path.getsize(p), "role": "primary"})
        for sc in resolve_sidecars(p):
            entries.append({"path": sc, "sha256": _sha256_file(sc), "n_bytes": os.path.getsize(sc), "role": "sidecar"})
    entries.sort(key=lambda e: e["path"])
    man = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"subject_dir": subject_dir, "files": entries, "manifest_sha256": man}
