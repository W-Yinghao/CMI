"""ACAR v3 structured data layer. DESIGN/DEV stage — SYNTHETIC contract only (no DEV cohort is read here).

Structured identity (Amendment 4): different datasets reuse local ids like 'sub-001', so all hash splits and
conformal grouping MUST key on the canonical SubjectKey(dataset_id, subject_id), never the bare subject_id.
    SubjectKey   = (dataset_id, subject_id)
    RecordingKey = (dataset_id, subject_id, recording_id)
    WindowKey    = (dataset_id, subject_id, recording_id, window_index)   [from set_features]

Type-level label firewall: the deployment path carries no labels; targets live in a separate Phase-2 type.
    DeploymentBatch  : window_keys, z, fallback, source_state_ref         — NO y
    LabeledRiskRecord: deployment_batch_digest, delta_r_by_action          — Phase-2 DEV target only
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import namedtuple, defaultdict
import hashlib
import json
import numpy as np

from acar.config import MIN_BATCH, B
from .set_features import WindowKey, NON_IDENTITY

SubjectKey = namedtuple("SubjectKey", "dataset_id subject_id")
RecordingKey = namedtuple("RecordingKey", "dataset_id subject_id recording_id")


def canon_subject(sk: SubjectKey) -> str:
    return "WS" + json.dumps([str(sk.dataset_id), str(sk.subject_id)], separators=(",", ":"))


def subject_of(wk: WindowKey) -> SubjectKey:
    return SubjectKey(wk.dataset_id, wk.subject_id)


def recording_of(wk: WindowKey) -> RecordingKey:
    return RecordingKey(wk.dataset_id, wk.subject_id, wk.recording_id)


@dataclass(frozen=True, slots=True)
class DeploymentBatch:
    subject: SubjectKey
    recording: RecordingKey
    window_keys: tuple
    z: np.ndarray
    fallback: bool
    source_state_ref: str

    def __post_init__(self):
        z = np.ascontiguousarray(np.asarray(self.z, float))
        if z.ndim != 2 or z.shape[0] != len(self.window_keys):
            raise ValueError("z must be [n_windows, d] aligned with window_keys")
        if not np.all(np.isfinite(z)):
            raise ValueError("non-finite z")
        if len(set(self.window_keys)) != len(self.window_keys):
            raise ValueError("duplicate window keys in batch")
        for wk in self.window_keys:
            if not isinstance(wk, WindowKey):
                raise TypeError("window_keys must be WindowKey")
            if subject_of(wk) != self.subject or recording_of(wk) != self.recording:
                raise ValueError("window key subject/recording mismatch")
        if not isinstance(self.source_state_ref, str) or not self.source_state_ref:
            raise ValueError("source_state_ref must be a non-empty str")
        z.flags.writeable = False
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "fallback", bool(self.fallback))


@dataclass(frozen=True, slots=True)
class LabeledRiskRecord:
    deployment_batch_digest: str
    delta_r_by_action: tuple        # ((action, float), ...) — Phase-2 DEV target

    def __post_init__(self):
        d = dict(self.delta_r_by_action)
        if set(d) != set(NON_IDENTITY):
            raise ValueError("delta_r_by_action must cover EXACTLY the non-identity actions")
        if not all(np.isfinite(v) for v in d.values()):
            raise ValueError("non-finite ΔR")
        if len(self.deployment_batch_digest) != 64:
            raise ValueError("deployment_batch_digest must be a full SHA-256")


def deployment_batch_digest(b: DeploymentBatch) -> str:
    h = hashlib.sha256()
    h.update(b"DB\x00"); h.update(canon_subject(b.subject).encode())
    h.update(json.dumps(list(map(str, b.recording)), separators=(",", ":")).encode())
    h.update(b.source_state_ref.encode()); h.update(np.array([b.fallback], np.uint8).tobytes())
    order = sorted(range(len(b.window_keys)), key=lambda i: int(b.window_keys[i].window_index))
    for i in order:
        wk = b.window_keys[i]
        h.update(json.dumps([str(wk.dataset_id), str(wk.subject_id), str(wk.recording_id), int(wk.window_index)],
                            separators=(",", ":")).encode())
        h.update(np.ascontiguousarray(b.z[i], dtype="<f8").tobytes())
    return h.hexdigest()


def build_deployment_batches(dataset_id, rows, source_state_ref, batch_size=B):
    """rows: iterable of (subject_id, recording_id, window_index, z_row). Window-ordered, recording-grouped, chunked.
    <MIN_BATCH chunks -> fallback=True (retained). Keys are canonical WindowKey; grouping is by SubjectKey/RecordingKey."""
    by_rec = defaultdict(list)
    for subj, rec, win, zr in rows:
        by_rec[(str(subj), str(rec))].append((int(win), np.asarray(zr, float)))
    out = []
    for (subj, rec) in sorted(by_rec):
        items = sorted(by_rec[(subj, rec)], key=lambda t: t[0])
        sk = SubjectKey(str(dataset_id), subj); rk = RecordingKey(str(dataset_id), subj, rec)
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            wks = tuple(WindowKey(str(dataset_id), subj, rec, w) for w, _ in chunk)
            z = np.stack([zr for _, zr in chunk], 0)
            out.append(DeploymentBatch(sk, rk, wks, z, len(chunk) < MIN_BATCH, source_state_ref))
    return out


def make_synthetic(n_datasets=2, subj_per=4, rec_per=1, win_per=20, d=8, seed=0):
    """Toy DeploymentBatches across datasets that REUSE local subject ids (to exercise SubjectKey disambiguation)."""
    rng = np.random.default_rng(seed); batches = []
    for di in range(n_datasets):
        ds = f"ds{di:03d}"
        rows = []
        for s in range(subj_per):
            for r in range(rec_per):
                for w in range(win_per):
                    rows.append((f"sub-{s:03d}", f"rec-{r:02d}", w, rng.standard_normal(d)))   # same local ids across ds
        batches += build_deployment_batches(ds, rows, source_state_ref=f"src::{ds}")
    return batches
