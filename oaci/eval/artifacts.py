"""Prediction artifacts + strict paired alignment.

A ``PredictionBundle`` is the saved output of one (method, seed, split, deletion_level) run.
Paired comparisons require IDENTICAL ``sample_id`` sets and identical ``y/domain/group/class``
maps — reordering by id is allowed, an inner-join that silently drops samples is NOT.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:16]


def population_hash(sample_id, y, domain, group) -> str:
    """Order-independent hash of the evaluation population (sorted by sample_id)."""
    order = np.argsort(np.asarray(sample_id))
    return _hash(np.asarray(sample_id)[order], np.asarray(y)[order],
                 np.asarray(domain)[order], np.asarray(group)[order])


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
        self.sample_id = np.asarray(self.sample_id)
        self.logits = np.asarray(self.logits, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=int).ravel()
        self.domain = np.asarray(self.domain, dtype=int).ravel()
        self.group = np.asarray(self.group, dtype=int).ravel()

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
