"""Immutable training population + its signature hash.

The batch plans reference rows by STABLE ``sample_id`` (never raw row index), so reordering the
input rows leaves the resolved training identical. ``population_signature_hash`` binds the id set,
labels, domain, group and the sample-mass bytes — but NOT the full EEG tensor (that is bound
separately by the data contract's content hash, to keep batch-plan hashing cheap).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TrainingData:
    X: torch.Tensor
    y: torch.Tensor
    sample_id: tuple
    sample_mass: torch.Tensor
    n_classes: int
    d: torch.Tensor | None = None
    group: tuple | None = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def validate(self) -> "TrainingData":
        n = len(self)
        if len(self.sample_id) != n:
            raise ValueError("sample_id length != X")
        if len(set(self.sample_id)) != n:
            raise ValueError("sample_id must be unique")
        if any((s is None or str(s) == "") for s in self.sample_id):
            raise ValueError("sample_id entries must be non-empty stable strings")
        for name, t in (("y", self.y), ("sample_mass", self.sample_mass)):
            if t.shape[0] != n:
                raise ValueError(f"{name} length != X")
        if self.d is not None and self.d.shape[0] != n:
            raise ValueError("d length != X")
        if self.group is not None and len(self.group) != n:
            raise ValueError("group length != X")
        if not torch.isfinite(self.X).all():
            raise ValueError("X has non-finite values")
        if not torch.isfinite(self.sample_mass).all() or float(self.sample_mass.min()) <= 0:
            raise ValueError("sample_mass must be finite and strictly positive")
        if int(self.y.min()) < 0 or int(self.y.max()) >= self.n_classes:
            raise ValueError("y out of [0, n_classes)")
        return self

    def index(self) -> dict:
        """sample_id -> row. A duplicate id is a hard error (a plan could not address it)."""
        idx: dict = {}
        for i, s in enumerate(self.sample_id):
            if s in idx:
                raise ValueError(f"duplicate sample_id in population: {s!r}")
            idx[s] = i
        return idx


def population_signature_hash(data: TrainingData) -> str:
    """Order-INVARIANT signature: canonical-sort by ``sample_id`` and bind id, y, d, group and the
    sample-mass dtype/bytes. Reordering input rows must NOT change this hash."""
    order = sorted(range(len(data)), key=lambda i: data.sample_id[i])
    h = hashlib.sha256()
    y = data.y.detach().cpu().numpy()
    d = None if data.d is None else data.d.detach().cpu().numpy()
    sm = data.sample_mass.detach().cpu().contiguous()
    h.update(str(sm.dtype).encode())
    for i in order:
        sid = str(data.sample_id[i]).encode()
        h.update(len(sid).to_bytes(8, "little")); h.update(sid)
        h.update(int(y[i]).to_bytes(8, "little", signed=True))
        h.update((b"-" if d is None else int(d[i]).to_bytes(8, "little", signed=True)))
        if data.group is None:                       # group: LENGTH-PREFIXED UTF-8 (match LeakageDesign)
            h.update(b"-")
        else:
            gb = str(data.group[i]).encode(); h.update(len(gb).to_bytes(8, "little")); h.update(gb)
        h.update(np.asarray(sm[i]).tobytes())
    return h.hexdigest()


def tensor_content_hash(X) -> str:
    """Content hash of the training feature tensor (kept SEPARATE from the batch-plan hash)."""
    t = (X.detach().cpu().contiguous() if isinstance(X, torch.Tensor)
         else torch.as_tensor(np.asarray(X)).contiguous())
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    for s in t.shape:
        h.update(int(s).to_bytes(8, "little"))
    h.update(t.numpy().tobytes())
    return h.hexdigest()
