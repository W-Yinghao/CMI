"""Leakage DESIGN — the label/domain/group/mass facts a leakage estimate depends on, with stable
hashes. It carries NO representation ``Z`` and NO method identity, so the same design (and the fold
/ bootstrap plans built from it) is reused byte-for-byte across every checkpoint and every method.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def hash_strings(h, strings) -> None:
    """Length-prefixed UTF-8 — never hash a NumPy object-array's pointer bytes."""
    for s in strings:
        b = str(s).encode("utf-8")
        h.update(len(b).to_bytes(8, "little")); h.update(b)


def _ro(a, dtype):
    a = np.array(np.asarray(a), dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


def population_hash(sample_id, y, d, group_id, sample_mass) -> str:
    """Row-order INVARIANT, but bound to each ID's (y, d, group, b_i). Strings length-prefixed."""
    sid = [str(s) for s in sample_id]
    order = sorted(range(len(sid)), key=lambda i: sid[i])
    h = hashlib.sha256()
    y = np.asarray(y); d = np.asarray(d); grp = [str(g) for g in group_id]
    mass = np.ascontiguousarray(np.asarray(sample_mass, dtype=np.float64))
    h.update(str(mass.dtype).encode())
    for i in order:
        hash_strings(h, [sid[i]])
        h.update(int(y[i]).to_bytes(8, "little", signed=True))
        h.update(int(d[i]).to_bytes(8, "little", signed=True))
        hash_strings(h, [grp[i]])
        h.update(np.asarray(mass[i]).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class LeakageDesign:
    sample_id: tuple
    y: np.ndarray
    d: np.ndarray
    group_id: tuple
    sample_mass: np.ndarray
    population_hash: str
    support_hash: str


def make_leakage_design(sample_id, y, d, group_id, sample_mass, support_graph) -> LeakageDesign:
    sid = tuple(str(s) for s in sample_id)
    grp = tuple(str(g) for g in group_id)
    yy = _ro(y, np.int64); dd = _ro(d, np.int64); mass = _ro(sample_mass, np.float64)
    n = len(sid)
    if not (len(grp) == n == yy.shape[0] == dd.shape[0] == mass.shape[0]):
        raise ValueError("design array lengths disagree")
    if len(set(sid)) != n or any(s == "" for s in sid):
        raise ValueError("sample_id must be unique and non-empty")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("sample_mass must be finite and strictly positive")
    # a recording group lives in exactly one domain
    g2d = {}
    for g, dv in zip(grp, dd.tolist()):
        if g in g2d and g2d[g] != int(dv):
            raise ValueError(f"group {g!r} spans domains {g2d[g]} and {int(dv)}")
        g2d[g] = int(dv)
    # the design's per-cell mass must agree with the support graph's cell_mass
    cm = np.zeros_like(support_graph.cell_mass, dtype=np.float64)
    for yi, di, mi in zip(yy.tolist(), dd.tolist(), mass.tolist()):
        cm[di, yi] += mi
    if not np.allclose(cm, support_graph.cell_mass, atol=1e-6):
        raise ValueError("design cell mass does not match SupportGraph.cell_mass")
    return LeakageDesign(sample_id=sid, y=yy, d=dd, group_id=grp, sample_mass=mass,
                         population_hash=population_hash(sid, yy, dd, grp, mass),
                         support_hash=support_graph.support_hash())
