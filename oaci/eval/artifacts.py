"""Prediction artifacts + strict paired alignment.

A ``PredictionBundle`` is the saved output of one (method, seed, split, deletion_level) run.
Paired comparisons require IDENTICAL ``sample_id`` sets and identical ``y/domain/group/class``
maps — reordering by id is allowed, an inner-join that silently drops samples is NOT.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _feed_strs(h, strs) -> None:
    """Length-prefixed UTF-8 — never hash a NumPy object-array's pointer bytes."""
    for s in strs:
        b = str(s).encode("utf-8"); h.update(len(b).to_bytes(8, "little")); h.update(b)


def _feed_arr(h, a) -> None:
    a = np.ascontiguousarray(np.asarray(a))
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())


def population_hash(sample_id, y, domain, group) -> str:
    """Order-INDEPENDENT hash of the evaluation population (sorted by string sample_id). sample_id
    and the STABLE STRING group are length-prefixed UTF-8; y / domain (a frozen contiguous int) are
    hashed as bytes. (An integer group stringifies, so integer-group inputs stay back-compatible.)"""
    sid = [str(s) for s in np.asarray(sample_id).tolist()]
    order = sorted(range(len(sid)), key=lambda i: sid[i])
    y = np.asarray(y).astype(np.int64); d = np.asarray(domain).astype(np.int64)
    g = [str(x) for x in np.asarray(group).tolist()]
    h = hashlib.sha256()
    for i in order:
        _feed_strs(h, [sid[i]])
        h.update(int(y[i]).to_bytes(8, "little", signed=True))
        h.update(int(d[i]).to_bytes(8, "little", signed=True))
        _feed_strs(h, [g[i]])
    return h.hexdigest()


@dataclass
class PredictionBundle:
    sample_id: np.ndarray
    logits: np.ndarray            # [N, C]
    y: np.ndarray
    domain: np.ndarray
    group: np.ndarray
    method: str
    seed: int
    split_id: str
    split_role: str               # 'source_guard' | 'source_audit' | 'target_audit'
    deletion_level: int
    class_names: list
    risk_metric: str = "balanced_ce"
    support_mask_hash: str = ""
    checkpoint_hash: str = ""
    audit_tensor_hash: str = ""        # hash of the actual audit X (identical IDs alone insufficient)
    split_manifest_hash: str = ""
    preprocess_hash: str = ""

    def __post_init__(self):
        self.sample_id = np.asarray([str(s) for s in np.asarray(self.sample_id).tolist()])
        self.logits = np.asarray(self.logits, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=int).ravel()
        self.domain = np.asarray(self.domain, dtype=int).ravel()                 # frozen contiguous int
        self.group = np.asarray([str(g) for g in np.asarray(self.group).ravel().tolist()], dtype=object)
        N = self.logits.shape[0]
        if N == 0:
            raise ValueError("prediction population is empty")
        if self.logits.ndim != 2:
            raise ValueError("logits must be [N, C]")
        C = self.logits.shape[1]
        if not np.all(np.isfinite(self.logits)):
            raise ValueError("logits must be finite")
        if not (len(self.sample_id) == self.y.shape[0] == self.domain.shape[0]
                == self.group.shape[0] == N):
            raise ValueError("sample_id / y / domain / group length disagree with logits")
        if len(set(self.sample_id.tolist())) != N or any(s == "" for s in self.sample_id.tolist()):
            raise ValueError("sample_id must be unique and non-empty")
        cn = list(self.class_names)
        if len(cn) == 0 or len(set(cn)) != len(cn) or any((not isinstance(c, str)) or c == "" for c in cn):
            raise ValueError("class_names must be non-empty, unique, non-empty strings")
        if C != len(cn):
            raise ValueError(f"logits second dim {C} != len(class_names) {len(cn)}")
        if int(self.y.min()) < 0 or int(self.y.max()) >= C:
            raise ValueError("y out of [0, C)")
        g2d = {}
        for g, dv in zip(self.group.tolist(), self.domain.tolist()):
            if g in g2d and g2d[g] != dv:
                raise ValueError(f"group {g} spans domains {g2d[g]} and {dv}")
            g2d[g] = dv
        if not str(self.method) or not str(self.split_id):
            raise ValueError("method and split_id must be non-empty")
        if self.split_role not in ("source_guard", "source_audit", "target_audit"):
            raise ValueError(f"invalid split_role {self.split_role!r}")
        if int(self.deletion_level) < 0:
            raise ValueError("deletion_level must be >= 0")

    @property
    def n(self) -> int:
        return self.logits.shape[0]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def pred(self) -> np.ndarray:
        return self.logits.argmax(axis=1)

    @property
    def eval_population_hash(self) -> str:
        return population_hash(self.sample_id, self.y, self.domain, self.group)

    def reorder(self, order) -> "PredictionBundle":
        order = np.asarray(order)
        return PredictionBundle(
            sample_id=self.sample_id[order], logits=self.logits[order], y=self.y[order],
            domain=self.domain[order], group=self.group[order], method=self.method, seed=self.seed,
            split_id=self.split_id, split_role=self.split_role, deletion_level=self.deletion_level,
            class_names=list(self.class_names), risk_metric=self.risk_metric,
            support_mask_hash=self.support_mask_hash, checkpoint_hash=self.checkpoint_hash,
            audit_tensor_hash=self.audit_tensor_hash, split_manifest_hash=self.split_manifest_hash,
            preprocess_hash=self.preprocess_hash,
        )

    def prediction_content_hash(self) -> str:
        """Byte identity of the prediction: sorted IDs + logits + labels + domain/group + class map +
        checkpoint hash + split metadata."""
        sid = [str(s) for s in self.sample_id.tolist()]
        order = sorted(range(len(sid)), key=lambda i: sid[i])
        h = hashlib.sha256()
        _feed_strs(h, [sid[i] for i in order])
        for arr in (self.logits[order], self.y[order], self.domain[order]):
            _feed_arr(h, arr)
        _feed_strs(h, [str(self.group[i]) for i in order])          # string group: never pointer-hash
        _feed_strs(h, list(self.class_names))
        _feed_strs(h, [self.method, self.split_id, self.split_role, str(int(self.deletion_level)),
                       self.checkpoint_hash, self.risk_metric])
        return h.hexdigest()

    def audit_signature(self) -> tuple:
        """Full byte-identity signature (population + actual tensor + split + preprocessing)."""
        return (self.eval_population_hash, self.audit_tensor_hash,
                self.split_manifest_hash, self.preprocess_hash)


def align_pair(a: PredictionBundle, b: PredictionBundle):
    """Sort both by ``sample_id`` and require identical id SETS and identical ``y/domain/group``
    and class map. Raises (never silently inner-joins/drops)."""
    sa, sb = a.sample_id, b.sample_id
    if sa.shape != sb.shape or set(sa.tolist()) != set(sb.tolist()):
        raise ValueError("paired comparison requires identical sample_id sets (no inner-join drop)")
    a2, b2 = a.reorder(np.argsort(sa)), b.reorder(np.argsort(sb))
    if not (np.array_equal(a2.sample_id, b2.sample_id) and np.array_equal(a2.y, b2.y)
            and np.array_equal(a2.domain, b2.domain) and np.array_equal(a2.group, b2.group)):
        raise ValueError("paired bundles disagree on y/domain/group at matched sample_id")
    if list(a2.class_names) != list(b2.class_names):
        raise ValueError("paired bundles disagree on the class map")
    return a2, b2
