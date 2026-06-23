"""Real-EEG data contract: ``EEGBundle`` with stable string IDs and the structural invariants
every loader must satisfy. IDs are stable strings (NOT load-order integers); ``sample_id`` is
unique; labels are constant within an ``eval_unit_id``; a ``group`` lives in exactly one domain.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def tensor_content_hash(X) -> str:
    return _sha(np.asarray(X, dtype=np.float32))     # full SHA-256 (real-data scientific identity)


@dataclass
class EEGBundle:
    X: np.ndarray                 # [N, C, T] float32
    y: np.ndarray                 # [N] int
    sample_id: np.ndarray         # [N] stable unique strings
    dataset_id: str
    site_id: np.ndarray           # [N] strings (cohort/site)
    subject_id: np.ndarray
    session_id: np.ndarray
    run_id: np.ndarray
    recording_id: np.ndarray      # the clustered-inference group key
    trial_id: np.ndarray
    support_unit_id: np.ndarray   # independent support unit (m-gate counts these)
    eval_unit_id: np.ndarray      # aggregation + one-label unit
    sfreq: float
    ch_names: list
    class_names: list
    preprocess_hash: str = ""
    raw_data_fingerprint: str = ""
    tensor_content_hash: str = ""

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=int).ravel()
        for f in ("sample_id", "site_id", "subject_id", "session_id", "run_id",
                  "recording_id", "trial_id", "support_unit_id", "eval_unit_id"):
            setattr(self, f, np.asarray(getattr(self, f), dtype=object).ravel())
        if not self.tensor_content_hash:
            self.tensor_content_hash = tensor_content_hash(self.X)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def validate(self) -> "EEGBundle":
        N = self.n
        for f in ("y", "sample_id", "site_id", "subject_id", "session_id", "run_id",
                  "recording_id", "trial_id", "support_unit_id", "eval_unit_id"):
            if len(getattr(self, f)) != N:
                raise ValueError(f"EEGBundle.{f} length {len(getattr(self,f))} != N {N}")
        if self.X.ndim != 3:
            raise ValueError(f"X must be [N,C,T], got {self.X.shape}")
        if not np.isfinite(self.X).all():
            raise ValueError("X contains non-finite values")
        if int(self.y.min()) < 0 or int(self.y.max()) >= len(self.class_names):
            raise ValueError("y out of class_names range")
        if np.unique(self.sample_id).size != N:
            raise ValueError("sample_id must be unique")
        if any(not isinstance(s, str) for s in self.sample_id[:1]):
            raise ValueError("sample_id must be strings (stable IDs, not load-order ints)")
        # one label per eval_unit
        for u in np.unique(self.eval_unit_id):
            if np.unique(self.y[self.eval_unit_id == u]).size != 1:
                raise ValueError(f"eval_unit_id {u!r} has more than one label")
        # a recording (group) lives in one domain (site) AND one subject
        for g in np.unique(self.recording_id):
            m = self.recording_id == g
            if np.unique(self.site_id[m]).size != 1 or np.unique(self.subject_id[m]).size != 1:
                raise ValueError(f"recording_id {g!r} spans multiple sites/subjects")
        return self

    def domain(self, factor: str = "site_id") -> np.ndarray:
        """Integer domain code per sample for a chosen factor (e.g. 'site_id' or 'subject_id')."""
        vals = getattr(self, factor)
        order = {v: i for i, v in enumerate(sorted(set(vals.tolist())))}
        return np.array([order[v] for v in vals], dtype=int)

    def group_codes(self) -> np.ndarray:
        order = {v: i for i, v in enumerate(sorted(set(self.recording_id.tolist())))}
        return np.array([order[v] for v in self.recording_id], dtype=int)
