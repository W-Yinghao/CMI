"""ACAR V5 Stage-1B typed reader payload (pure/stdlib at import). `read_subject_windows` returns a SubjectWindows (SIGNAL ONLY — no
label); labels are a separate FIT-only read. The payload now carries the ACTUAL window array (`windows`, shape
[n_windows, 19, 512], floating, finite), and the schema is fail-closed against the pinned preprocessing config (exact 19-channel
order, 128 Hz, 512-sample windows). numpy is imported LAZILY inside the payload validator only — importing this module pulls no
heavy dependency.
"""
from __future__ import annotations
from dataclasses import dataclass
from acar.v5.substrate import preprocessing_config as PC


class SubjectWindowsError(RuntimeError):
    pass


@dataclass(frozen=True)
class SubjectWindows:
    subject_key: str
    disease: str
    cohort: str
    raw_subject_id: str
    n_windows: int
    n_channels: int
    n_samples: int
    sfreq: int
    channels: tuple
    preprocessing_config_sha256: str
    windows: object = None                 # array-like [n_windows, 19, 512] float; duck-typed, validated lazily
    provenance: str = ""
    montage_completion: object = None      # {interpolated:[...], n_interpolated, donor_count, by_recording:[...]} or None
    read_repair: object = None             # Stage-1B12: {repaired:[recording...], by_recording:[repair manifest...]} or None
    # NOTE: no `label` / `y` field here — labels are read separately and ONLY via the FIT training view.


def validate_subject_windows(sw):
    """Fail-closed schema check against the pinned preprocessing config, INCLUDING the actual window payload."""
    if not isinstance(sw, SubjectWindows):
        raise SubjectWindowsError("payload must be a SubjectWindows")
    if "/" in sw.raw_subject_id:
        raise SubjectWindowsError(f"raw_subject_id must be un-namespaced: {sw.raw_subject_id!r}")
    if sw.subject_key != f"{sw.disease}/{sw.cohort}/{sw.raw_subject_id}":
        raise SubjectWindowsError(f"subject_key {sw.subject_key!r} inconsistent with disease/cohort/raw")
    if tuple(sw.channels) != PC.CHANNELS_19:
        raise SubjectWindowsError("channels must be EXACTLY the pinned 19-channel order")
    exp_hz, exp_samp = PC.PREPROCESSING_CONFIG["resample_hz"], PC.PREPROCESSING_CONFIG["window_samples"]
    if sw.n_channels != 19 or sw.sfreq != exp_hz or sw.n_samples != exp_samp:
        raise SubjectWindowsError(f"n_channels/sfreq/n_samples must be 19/{exp_hz}/{exp_samp}")
    if not isinstance(sw.n_windows, int) or sw.n_windows < 1:
        raise SubjectWindowsError("n_windows must be a positive int")
    if sw.preprocessing_config_sha256 != PC.config_sha256():
        raise SubjectWindowsError("preprocessing_config_sha256 mismatch vs the pinned config")
    _validate_payload(sw)
    return True


def _validate_payload(sw):
    """Duck-typed payload check: shape == (n_windows, 19, 512), floating dtype, finite (no NaN/Inf). numpy imported lazily."""
    w = sw.windows
    if w is None:
        raise SubjectWindowsError("windows payload is required (SubjectWindows must carry the actual signal array)")
    shape = getattr(w, "shape", None)
    if shape is None:
        raise SubjectWindowsError("windows must be an array-like with a .shape")
    if tuple(shape) != (sw.n_windows, sw.n_channels, sw.n_samples):
        raise SubjectWindowsError(f"windows shape {tuple(shape)} != (n_windows,n_channels,n_samples) "
                                  f"({sw.n_windows},{sw.n_channels},{sw.n_samples})")
    dtype = getattr(w, "dtype", None)
    if dtype is None or getattr(dtype, "kind", None) != "f":
        raise SubjectWindowsError("windows dtype must be floating")
    import numpy as np  # lazy — never imported at module load
    if not bool(np.isfinite(w).all()):
        raise SubjectWindowsError("windows must be finite (no NaN / Inf)")
    return True


def has_label_field(obj):
    """True if an object/dict leaks a label-like field (used by the label-free embedding-dump guard)."""
    forbidden = ("label", "y", "y_te", "diagnosis", "target", "case_control", "labels")
    keys = obj.keys() if isinstance(obj, dict) else [a for a in dir(obj) if not a.startswith("__")]
    return any(k in forbidden for k in keys)
