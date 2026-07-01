"""ACAR V5 Stage-1B typed reader payload (pure/stdlib). `read_subject_windows` returns a SubjectWindows (SIGNAL ONLY — no label);
labels are a separate FIT-only read. Schema is fail-closed against the pinned preprocessing config (exact 19-channel order, 128 Hz,
512-sample windows).
"""
from __future__ import annotations
from dataclasses import dataclass, field
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
    provenance: str = ""
    # NOTE: no `label` / `y` field here — labels are read separately and ONLY via the FIT training view.


def validate_subject_windows(sw):
    """Fail-closed schema check against the pinned preprocessing config."""
    if not isinstance(sw, SubjectWindows):
        raise SubjectWindowsError("payload must be a SubjectWindows")
    if "/" in sw.raw_subject_id:
        raise SubjectWindowsError(f"raw_subject_id must be un-namespaced: {sw.raw_subject_id!r}")
    if sw.subject_key != f"{sw.disease}/{sw.cohort}/{sw.raw_subject_id}":
        raise SubjectWindowsError(f"subject_key {sw.subject_key!r} inconsistent with disease/cohort/raw")
    if tuple(sw.channels) != PC.CHANNELS_19:
        raise SubjectWindowsError("channels must be EXACTLY the pinned 19-channel order")
    if sw.n_channels != 19 or sw.sfreq != PC.PREPROCESSING_CONFIG["resample_hz"] or sw.n_samples != PC.PREPROCESSING_CONFIG["window_samples"]:
        raise SubjectWindowsError(f"n_channels/sfreq/n_samples must be 19/{PC.PREPROCESSING_CONFIG['resample_hz']}/{PC.PREPROCESSING_CONFIG['window_samples']}")
    if not isinstance(sw.n_windows, int) or sw.n_windows < 1:
        raise SubjectWindowsError("n_windows must be a positive int")
    if sw.preprocessing_config_sha256 != PC.config_sha256():
        raise SubjectWindowsError("preprocessing_config_sha256 mismatch vs the pinned config")
    return True


def has_label_field(obj):
    """True if an object/dict leaks a label-like field (used by the label-free embedding-dump guard)."""
    forbidden = ("label", "y", "y_te", "diagnosis", "target", "case_control", "labels")
    keys = obj.keys() if isinstance(obj, dict) else [a for a in dir(obj) if not a.startswith("__")]
    return any(k in forbidden for k in keys)
