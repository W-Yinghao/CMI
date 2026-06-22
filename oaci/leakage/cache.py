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


def _read_only(result: dict) -> dict:
    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            vv = np.array(v, copy=True); vv.setflags(write=False); out[k] = vv
        else:
            out[k] = v
    return out


class LeakageScoreCache:
    def __init__(self):
        self._store: dict = {}
        self._counts: dict = defaultdict(int)

    def get(self, key: LeakageScoreKey):
        return self._store.get(key)

    def put(self, key: LeakageScoreKey, result: dict) -> None:
        self._store[key] = _read_only(result)

    def get_or_compute(self, key: LeakageScoreKey, fn):
        if key in self._store:
            return self._store[key]
        self._counts[key] += 1
        self._store[key] = _read_only(fn())
        return self._store[key]

    def compute_count(self, key: LeakageScoreKey) -> int:
        return int(self._counts[key])
