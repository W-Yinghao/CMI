"""Leakage score cache — the ERM score for one deletion level is extracted/probed/bootstrapped ONCE
and read by all three Stage-2 selectors.

The key binds the ACTUAL frozen-feature bytes (a model hash + sample population is NOT enough to
prove the input tensor is identical) and the critic-config hash. Cached arrays are read-only so a
caller cannot mutate the shared result.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def frozen_feature_hash(Z) -> str:
    a = np.ascontiguousarray(np.asarray(Z))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def critic_config_hash(cfg) -> str:
    h = hashlib.sha256()
    h.update(str(tuple(int(c) for c in cfg.capacities)).encode())
    for v in (cfg.l2_C, cfg.max_iter, cfg.prob_floor, cfg.feature_seed_base):
        h.update(f"{v!r}".encode())
    return h.hexdigest()


@dataclass(frozen=True)
class LeakageScoreKey:
    model_hash: str
    frozen_feature_hash: str
    population_hash: str
    support_hash: str
    fold_plan_hash: str
    bootstrap_plan_hash: str
    critic_config_hash: str


def _deep_freeze(v):
    """Defensive deep copy with every NumPy array set read-only (recurses into dict/list/tuple)."""
    if isinstance(v, np.ndarray):
        a = np.array(v, copy=True); a.setflags(write=False); return a
    if isinstance(v, dict):
        return {k: _deep_freeze(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_deep_freeze(x) for x in v)
    return v


class LeakageScoreCache:
    def __init__(self):
        self._store: dict = {}
        self._compute: dict = defaultdict(int)
        self._request: dict = defaultdict(int)
        self._hit: dict = defaultdict(int)

    def get(self, key: LeakageScoreKey):
        return self._store.get(key)

    def put(self, key: LeakageScoreKey, result: dict) -> None:
        self._store[key] = _deep_freeze(result)

    def get_or_compute(self, key: LeakageScoreKey, fn):
        self._request[key] += 1
        if key in self._store:
            self._hit[key] += 1
            return self._store[key]
        self._compute[key] += 1
        self._store[key] = _deep_freeze(fn())
        return self._store[key]

    def request_count(self, key: LeakageScoreKey) -> int:
        return int(self._request[key])

    def compute_count(self, key: LeakageScoreKey) -> int:
        return int(self._compute[key])

    def hit_count(self, key: LeakageScoreKey) -> int:
        return int(self._hit[key])

    def total_requests(self) -> int:
        return int(sum(self._request.values()))

    def total_computes(self) -> int:
        return int(sum(self._compute.values()))

    def total_hits(self) -> int:
        return int(sum(self._hit.values()))
