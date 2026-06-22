"""ACAR v3 structured data layer. DESIGN/DEV stage — SYNTHETIC contract only (no DEV cohort is read here).

Structured identity (Amendment 4): hash splits + conformal grouping key on canonical SubjectKey(dataset_id,
subject_id) — different datasets reuse 'sub-001' and must never merge.
    SubjectKey/RecordingKey/WindowKey ; ids non-empty, window_index int >= 0 (validated at the contract boundary).

Type-level label firewall + batching-protocol validation:
    DeploymentBatch  : disease, subject, recording, window_keys, z, fallback, source_state_ref  — NO y.
                       fallback <=> n_windows < MIN_BATCH; 1 <= n_windows <= B; source_state_ref is a 64-hex SHA-256.
    LabeledRiskRecord: deployment_batch_digest, delta_r_by_action (canonical order, full-hex digest) — Phase-2 only.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
import numpy as np

from acar.config import MIN_BATCH, B
from .set_features import WindowKey, NON_IDENTITY
from ._util import frozen_array

DATA_SCHEMA = "acar-v3-data/1"


@dataclass(frozen=True, slots=True)
class SubjectKey:
    dataset_id: str
    subject_id: str

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.dataset_id, self.subject_id)):
            raise ValueError("SubjectKey ids must be non-empty str (no coercion)")


@dataclass(frozen=True, slots=True)
class RecordingKey:
    dataset_id: str
    subject_id: str
    recording_id: str

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.dataset_id, self.subject_id, self.recording_id)):
            raise ValueError("RecordingKey ids must be non-empty str (no coercion)")


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)   # lowercase only


def canon_subject(sk: SubjectKey) -> str:
    return "WS" + json.dumps([str(sk.dataset_id), str(sk.subject_id)], separators=(",", ":"))


def subject_of(wk: WindowKey) -> SubjectKey:
    return SubjectKey(wk.dataset_id, wk.subject_id)


def recording_of(wk: WindowKey) -> RecordingKey:
    return RecordingKey(wk.dataset_id, wk.subject_id, wk.recording_id)


def _validate_window_key(wk):
    if not isinstance(wk, WindowKey):
        raise TypeError("window key must be WindowKey")
    for f in (wk.dataset_id, wk.subject_id, wk.recording_id):
        if not isinstance(f, str) or f == "":
            raise ValueError("empty id component in WindowKey")
    if not isinstance(wk.window_index, (int, np.integer)) or int(wk.window_index) < 0:
        raise ValueError("window_index must be a non-negative int")


@dataclass(frozen=True, slots=True)
class DeploymentBatch:
    disease: str
    subject: SubjectKey
    recording: RecordingKey
    window_keys: tuple
    z: np.ndarray
    fallback: bool
    source_state_ref: str

    def __post_init__(self):
        if self.disease not in ("PD", "SCZ"):
            raise ValueError("disease must be PD or SCZ")
        if not isinstance(self.subject, SubjectKey) or not isinstance(self.recording, RecordingKey):
            raise TypeError("subject/recording must be SubjectKey/RecordingKey (no plain tuples)")
        keys = tuple(self.window_keys)
        z = np.ascontiguousarray(np.asarray(self.z, float))
        n = len(keys)
        if z.ndim != 2 or z.shape[0] != n:
            raise ValueError("z must be [n_windows, d] aligned with window_keys")
        if not (1 <= n <= B):
            raise ValueError(f"n_windows must be in [1, {B}] (got {n})")
        if z.shape[1] < 1:
            raise ValueError("embedding dimension d must be >= 1")
        if not np.all(np.isfinite(z)):
            raise ValueError("non-finite z")
        if not isinstance(self.fallback, bool):
            raise TypeError("fallback must be a bool")
        if self.fallback != (n < MIN_BATCH):
            raise ValueError(f"fallback ({self.fallback}) must equal n_windows<{MIN_BATCH} ({n < MIN_BATCH})")
        if not _is_hex64(self.source_state_ref):
            raise ValueError("source_state_ref must be a 64-char hex SHA-256")
        seen = set()
        for wk in keys:
            _validate_window_key(wk)
            if subject_of(wk) != self.subject or recording_of(wk) != self.recording:
                raise ValueError("window key subject/recording mismatch")
            if wk in seen:
                raise ValueError("duplicate window key")
            seen.add(wk)
        object.__setattr__(self, "window_keys", keys)
        object.__setattr__(self, "z", frozen_array(z))


@dataclass(frozen=True, slots=True)
class LabeledRiskRecord:
    deployment_batch_digest: str
    delta_r_by_action: tuple        # MUST be in canonical NON_IDENTITY order

    def __post_init__(self):
        if not _is_hex64(self.deployment_batch_digest):
            raise ValueError("deployment_batch_digest must be a full hex SHA-256")
        items = tuple((a, float(v)) for a, v in self.delta_r_by_action)
        if tuple(a for a, _ in items) != NON_IDENTITY:
            raise ValueError(f"delta_r_by_action must be in canonical order {NON_IDENTITY}")
        if any(not np.isfinite(v) for _, v in items):
            raise ValueError("non-finite ΔR")
        object.__setattr__(self, "delta_r_by_action", items)


def deployment_batch_digest(b: DeploymentBatch) -> str:
    h = hashlib.sha256()
    head = json.dumps({"schema": DATA_SCHEMA, "disease": b.disease, "subject": canon_subject(b.subject),
                       "recording": [str(b.recording.dataset_id), str(b.recording.subject_id), str(b.recording.recording_id)],
                       "n_windows": int(b.z.shape[0]), "d": int(b.z.shape[1]), "dtype": str(b.z.dtype),
                       "fallback": bool(b.fallback), "source_state_ref": b.source_state_ref}, sort_keys=True).encode()
    h.update(b"DBHDR\x00"); h.update(head)
    order = sorted(range(len(b.window_keys)), key=lambda i: int(b.window_keys[i].window_index))
    for i in order:
        wk = b.window_keys[i]
        h.update(json.dumps([str(wk.dataset_id), str(wk.subject_id), str(wk.recording_id), int(wk.window_index)],
                            separators=(",", ":")).encode())
        h.update(np.ascontiguousarray(b.z[i], dtype="<f8").tobytes())
    return h.hexdigest()


def build_deployment_batches(dataset_id, disease, rows, source_state_ref):
    """rows: (subject_id:str, recording_id:str, window_index:int, z_row). Window-ordered, recording-grouped, chunked
    at the FROZEN B=32 (no alternative batching). NO str/int coercion (so 1 and "1" never merge). Window indices
    MUST be unique within a recording (checked before chunking). All z rows must share one embedding dimension."""
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty str")
    rows = list(rows)
    if not rows:
        raise ValueError("empty rows")
    by_rec = defaultdict(list); dims = set()
    for subj, rec, win, zr in rows:
        if not (isinstance(subj, str) and subj and isinstance(rec, str) and rec):
            raise ValueError("subject_id/recording_id must be non-empty str (no coercion)")
        if isinstance(win, bool) or not isinstance(win, int) or win < 0:
            raise ValueError("window_index must be a non-negative int (no bool/coercion)")
        z = np.asarray(zr, float)
        if z.ndim != 1:
            raise ValueError("z_row must be 1-D")
        dims.add(int(z.shape[0]))
        by_rec[(subj, rec)].append((win, z))
    if len(dims) != 1:
        raise ValueError(f"inconsistent embedding dimension across rows: {sorted(dims)}")
    out = []
    for (subj, rec) in sorted(by_rec):
        items = sorted(by_rec[(subj, rec)], key=lambda t: t[0])
        idx = [w for w, _ in items]
        if len(set(idx)) != len(idx):
            raise ValueError(f"duplicate window_index within recording {(dataset_id, subj, rec)}")
        sk = SubjectKey(dataset_id, subj); rk = RecordingKey(dataset_id, subj, rec)
        for s in range(0, len(items), B):                    # FROZEN B
            chunk = items[s:s + B]
            wks = tuple(WindowKey(dataset_id, subj, rec, w) for w, _ in chunk)
            z = np.stack([zr for _, zr in chunk], 0)
            out.append(DeploymentBatch(disease, sk, rk, wks, z, len(chunk) < MIN_BATCH, source_state_ref))
    return out


def make_synthetic(n_datasets=2, subj_per=4, rec_per=1, win_per=20, d=8, disease="PD", seed=0):
    """Toy DeploymentBatches across datasets that REUSE local subject ids (exercises SubjectKey disambiguation)."""
    rng = np.random.default_rng(seed); batches = []
    for di in range(n_datasets):
        ds = f"ds{di:03d}"
        src = hashlib.sha256(f"src::{ds}".encode()).hexdigest()
        rows = [(f"sub-{s:03d}", f"rec-{r:02d}", w, rng.standard_normal(d))
                for s in range(subj_per) for r in range(rec_per) for w in range(win_per)]
        batches += build_deployment_batches(ds, disease, rows, src)
    return batches
