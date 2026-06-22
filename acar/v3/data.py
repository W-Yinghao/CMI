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
from collections import namedtuple, defaultdict
import hashlib
import json
import string
import numpy as np

from acar.config import MIN_BATCH, B
from .set_features import WindowKey, NON_IDENTITY

DATA_SCHEMA = "acar-v3-data/1"
SubjectKey = namedtuple("SubjectKey", "dataset_id subject_id")
RecordingKey = namedtuple("RecordingKey", "dataset_id subject_id recording_id")


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(c in string.hexdigits for c in s)


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
        z = np.ascontiguousarray(np.asarray(self.z, float))
        n = len(self.window_keys)
        if z.ndim != 2 or z.shape[0] != n:
            raise ValueError("z must be [n_windows, d] aligned with window_keys")
        if not (1 <= n <= B):
            raise ValueError(f"n_windows must be in [1, {B}] (got {n})")
        if not np.all(np.isfinite(z)):
            raise ValueError("non-finite z")
        if bool(self.fallback) != (n < MIN_BATCH):
            raise ValueError(f"fallback ({self.fallback}) must equal n_windows<{MIN_BATCH} ({n < MIN_BATCH})")
        if not _is_hex64(self.source_state_ref):
            raise ValueError("source_state_ref must be a 64-char hex SHA-256")
        seen = set()
        for wk in self.window_keys:
            _validate_window_key(wk)
            if subject_of(wk) != self.subject or recording_of(wk) != self.recording:
                raise ValueError("window key subject/recording mismatch")
            if wk in seen:
                raise ValueError("duplicate window key")
            seen.add(wk)
        z.flags.writeable = False
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "fallback", bool(self.fallback))


@dataclass(frozen=True, slots=True)
class LabeledRiskRecord:
    deployment_batch_digest: str
    delta_r_by_action: tuple        # MUST be in canonical NON_IDENTITY order

    def __post_init__(self):
        if not _is_hex64(self.deployment_batch_digest):
            raise ValueError("deployment_batch_digest must be a full hex SHA-256")
        acts = tuple(a for a, _ in self.delta_r_by_action)
        if acts != NON_IDENTITY:
            raise ValueError(f"delta_r_by_action must be in canonical order {NON_IDENTITY}; got {acts}")
        if any(not np.isfinite(v) for _, v in self.delta_r_by_action):
            raise ValueError("non-finite ΔR")


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


def build_deployment_batches(dataset_id, disease, rows, source_state_ref, batch_size=B):
    """rows: (subject_id, recording_id, window_index, z_row). Window-ordered, recording-grouped, chunked. Window
    indices MUST be unique within a recording (checked before chunking, so a duplicate cannot hide across a chunk
    boundary). fallback derived from chunk size. Keys canonical WindowKey; grouping by SubjectKey/RecordingKey."""
    by_rec = defaultdict(list)
    for subj, rec, win, zr in rows:
        by_rec[(str(subj), str(rec))].append((int(win), np.asarray(zr, float)))
    out = []
    for (subj, rec) in sorted(by_rec):
        items = sorted(by_rec[(subj, rec)], key=lambda t: t[0])
        idx = [w for w, _ in items]
        if len(set(idx)) != len(idx):
            raise ValueError(f"duplicate window_index within recording {(dataset_id, subj, rec)}")
        sk = SubjectKey(str(dataset_id), subj); rk = RecordingKey(str(dataset_id), subj, rec)
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            wks = tuple(WindowKey(str(dataset_id), subj, rec, w) for w, _ in chunk)
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
