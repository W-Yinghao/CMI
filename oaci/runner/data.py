"""FoldData — the 3-role (source_train / source_audit / target_audit) data contract.

Built ONLY via ``FoldData.from_arrays``, which defensive-copies (X cloned, arrays read-only),
validates the role split and the unit nesting, and computes every subset population/tensor hash
itself (callers cannot pre-fill scientific identity). A recording group MAY span classes; a
support/mass/eval unit may not span domain/class/group/role. ``assert_integrity`` re-checks the
tensor afterwards (a frozen dataclass cannot stop in-place Torch mutation).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .keys import feed_float64, feed_int64, feed_string

_MASS_TOL = 1e-9


def _ro(a, dtype):
    a = np.array(np.asarray(a), dtype=dtype, copy=True); a.setflags(write=False); return a


def _strs(seq):
    return tuple(str(s) for s in seq)


def _subset_population_hash(rows, role, sid, y, dom, grp, su, mu, eu, mass, class_names, prep, split):
    rows = sorted(rows, key=lambda i: sid[i])
    h = hashlib.sha256()
    feed_string(h, role)
    for c in class_names:
        feed_string(h, c)
    feed_string(h, prep); feed_string(h, split)
    for i in rows:
        for s in (sid[i], dom[i], grp[i], su[i], mu[i], eu[i]):
            feed_string(h, s)
        feed_int64(h, int(y[i])); feed_float64(h, float(mass[i]))
    return h.hexdigest()


def _subset_tensor_hash(rows, pop_hash, X, sid):
    rows = sorted(rows, key=lambda i: sid[i])
    h = hashlib.sha256(); h.update(pop_hash.encode())
    h.update(str(X.dtype).encode()); h.update(str(tuple(X.shape[1:])).encode())
    for i in rows:
        h.update(np.ascontiguousarray(X[i].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FoldData:
    X: torch.Tensor
    y: np.ndarray
    sample_id: tuple
    domain_id: tuple
    group_id: tuple
    support_unit_id: tuple
    mass_unit_id: tuple
    eval_unit_id: tuple
    sample_mass: np.ndarray
    class_names: tuple
    source_train_idx: np.ndarray
    source_audit_idx: np.ndarray
    target_audit_idx: np.ndarray
    preprocess_hash: str
    split_manifest_hash: str
    preprocess_fit_ids: frozenset
    source_train_population_hash: str
    source_audit_population_hash: str
    source_audit_tensor_hash: str
    target_population_hash: str
    target_tensor_hash: str
    data_contract_hash: str
    _integrity_hash: str

    def role_ids(self, role: str) -> tuple:
        idx = {"source_train": self.source_train_idx, "source_audit": self.source_audit_idx,
               "target_audit": self.target_audit_idx}[role]
        return tuple(self.sample_id[i] for i in idx.tolist())

    def assert_integrity(self) -> None:
        if _subset_tensor_hash(range(len(self.sample_id)), "all", self.X, self.sample_id) != self._integrity_hash:
            raise ValueError("FoldData tensor was mutated after construction")

    @staticmethod
    def from_arrays(X, y, sample_id, domain_id, group_id, support_unit_id, mass_unit_id, eval_unit_id,
                    sample_mass, class_names, source_train_idx, source_audit_idx, target_audit_idx,
                    preprocess_hash, split_manifest_hash, preprocess_fit_ids) -> "FoldData":
        X = X.detach().cpu().clone().contiguous()
        y = _ro(y, np.int64); mass = _ro(sample_mass, np.float64)
        sid, dom, grp = _strs(sample_id), _strs(domain_id), _strs(group_id)
        su, mu, eu = _strs(support_unit_id), _strs(mass_unit_id), _strs(eval_unit_id)
        cn = tuple(str(c) for c in class_names)
        n = int(X.shape[0])
        for name, seq in (("y", y), ("sample_mass", mass), ("sample_id", sid), ("domain_id", dom),
                          ("group_id", grp), ("support_unit_id", su), ("mass_unit_id", mu), ("eval_unit_id", eu)):
            if len(seq) != n:
                raise ValueError(f"{name} length {len(seq)} != X rows {n}")
        if len(set(sid)) != n or any(s == "" for s in sid):
            raise ValueError("sample_id must be unique and non-empty")
        for nm, seq in (("domain_id", dom), ("group_id", grp), ("support_unit_id", su),
                        ("mass_unit_id", mu), ("eval_unit_id", eu)):
            if any(s == "" for s in seq):
                raise ValueError(f"{nm} has an empty id")
        if len(cn) == 0 or len(set(cn)) != len(cn) or any(c == "" for c in cn):
            raise ValueError("class_names must be non-empty, unique, and individually non-empty")
        if not str(preprocess_hash) or not str(split_manifest_hash):
            raise ValueError("preprocess_hash and split_manifest_hash must be non-empty")
        if not torch.isfinite(X).all() or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
            raise ValueError("X/sample_mass must be finite and mass strictly positive")
        if int(y.min()) < 0 or int(y.max()) >= len(cn):
            raise ValueError("y out of [0, n_classes)")

        st = _ro(np.unique(source_train_idx), np.int64)
        sa = _ro(np.unique(source_audit_idx), np.int64)
        ta = _ro(np.unique(target_audit_idx), np.int64)
        for nm, r in (("source_train", st), ("source_audit", sa), ("target_audit", ta)):
            if r.size == 0:
                raise ValueError(f"role {nm} is empty")
        sets = [set(st.tolist()), set(sa.tolist()), set(ta.tolist())]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("roles overlap")
        if sets[0] | sets[1] | sets[2] != set(range(n)):
            raise ValueError("every row must belong to exactly one role")
        role = np.empty(n, dtype=object)
        for r, nm in ((st, "source_train"), (sa, "source_audit"), (ta, "target_audit")):
            for i in r.tolist():
                role[i] = nm

        # group spans ONE (domain, role) but MAY span classes
        g2dr = {}
        for i in range(n):
            k = (dom[i], role[i])
            if grp[i] in g2dr and g2dr[grp[i]] != k:
                raise ValueError(f"group {grp[i]!r} spans domains/roles {g2dr[grp[i]]} and {k}")
            g2dr[grp[i]] = k
        # support/mass/eval units span ONE (domain, class, group, role)
        for ids, label in ((su, "support_unit"), (mu, "mass_unit"), (eu, "eval_unit")):
            u2k = {}
            for i in range(n):
                k = (dom[i], int(y[i]), grp[i], role[i])
                if ids[i] in u2k and u2k[ids[i]] != k:
                    raise ValueError(f"{label} {ids[i]!r} spans cells {u2k[ids[i]]} and {k}")
                u2k[ids[i]] = k
        # mass unit base mass sums to 1
        umass = {}
        for i in range(n):
            umass[mu[i]] = umass.get(mu[i], 0.0) + float(mass[i])
        for u, m in umass.items():
            if abs(m - 1.0) > _MASS_TOL:
                raise ValueError(f"mass unit {u!r} base mass {m} != 1")
        # source_train: every pre-registered class has positive mass
        st_set = sets[0]
        for c in range(len(cn)):
            if not any((role[i] == "source_train" and int(y[i]) == c) for i in range(n)) or \
               sum(float(mass[i]) for i in range(n) if role[i] == "source_train" and int(y[i]) == c) <= 0:
                raise ValueError(f"source_train class {cn[c]} has zero mass")
        fit = frozenset(str(s) for s in preprocess_fit_ids)
        st_ids = {sid[i] for i in st_set}
        if not fit <= st_ids:
            raise ValueError("preprocess_fit_ids not subset of source_train ids")

        def pop(rows, rname):
            return _subset_population_hash(rows, rname, sid, y, dom, grp, su, mu, eu, mass, cn,
                                           preprocess_hash, split_manifest_hash)
        st_pop = pop(st.tolist(), "source_train")
        sa_pop = pop(sa.tolist(), "source_audit"); sa_t = _subset_tensor_hash(sa.tolist(), sa_pop, X, sid)
        ta_pop = pop(ta.tolist(), "target_audit"); ta_t = _subset_tensor_hash(ta.tolist(), ta_pop, X, sid)
        integ = _subset_tensor_hash(range(n), "all", X, sid)
        dch = hashlib.sha256("|".join([st_pop, sa_pop, sa_t, ta_pop, ta_t, preprocess_hash,
                                       split_manifest_hash, ",".join(cn)]).encode()).hexdigest()
        return FoldData(X, y, sid, dom, grp, su, mu, eu, mass, cn, st, sa, ta, preprocess_hash,
                        split_manifest_hash, fit, st_pop, sa_pop, sa_t, ta_pop, ta_t, dch, integ)
