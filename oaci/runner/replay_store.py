"""Replay store for GPU-extracted artifacts (C4b foundation).

The only device-dependent work in selection / audit / finalize is the forward pass that produces the
frozen features (``extract_frozen_features``) and the row predictions (``predict_checkpoint``). Both sit
behind deterministic keys (``FeatureArtifactKey`` / ``PredictionCacheKey``). This module lets a GPU run
RECORD those exact outputs and a later run REPLAY them -- so the CPU-bound leakage/eval scoring can run
without holding the GPU, and bit-exactly (the replayed bytes are the SAME GPU-extracted bytes).

Ambient modes (process-wide; never enters any hash):
* ``off``    (default) -- no store interaction; the existing behaviour.
* ``record`` -- every freshly computed artifact is stored under its key.
* ``replay`` -- every artifact is served from the store; a MISSING key is a hard error (so a successful
  replay proves the store fully captured the GPU forwards) and the compute function is never called.
"""
from __future__ import annotations

import pickle

_AMBIENT = {"store": None, "mode": "off"}
_MODES = ("off", "record", "replay")


class ReplayStore:
    """{(kind, key) -> artifact}. Keys are the frozen cache-key dataclasses; values are the numpy-backed
    FeatureArtifact / RowPredictionArtifact -- all picklable, so the store is persistable between jobs."""

    def __init__(self):
        self._d = {}

    def record(self, kind: str, key, value) -> None:
        self._d[(kind, key)] = value

    def has(self, kind: str, key) -> bool:
        return (kind, key) in self._d

    def lookup(self, kind: str, key):
        if (kind, key) not in self._d:
            raise KeyError(f"replay store has no {kind} artifact for key {key!r} "
                           f"(the GPU record stage did not extract it)")
        return self._d[(kind, key)]

    def drop(self, kind: str, key) -> None:
        self._d.pop((kind, key), None)

    def kinds(self) -> dict:
        out = {}
        for (kind, _key) in self._d:
            out[kind] = out.get(kind, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self._d)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._d, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "ReplayStore":
        s = cls()
        with open(path, "rb") as f:
            s._d = pickle.load(f)
        return s


def set_replay_store(store, mode: str = "off") -> None:
    if mode not in _MODES:
        raise ValueError(f"replay mode must be one of {_MODES}; got {mode!r}")
    if mode in ("record", "replay") and store is None:
        raise ValueError(f"mode {mode!r} requires a ReplayStore")
    _AMBIENT["store"] = store
    _AMBIENT["mode"] = mode


def get_replay_store():
    return _AMBIENT["store"]


def replay_mode() -> str:
    return _AMBIENT["mode"]


def resolve_artifact(kind: str, key, compute_fn):
    """The single hook the feature/prediction caches call on a local-cache MISS. In ``off`` mode it just
    computes (the existing behaviour); in ``record`` it computes and stores; in ``replay`` it serves the
    stored artifact and NEVER computes (a missing key raises)."""
    mode = _AMBIENT["mode"]
    store = _AMBIENT["store"]
    if mode == "off" or store is None:
        return compute_fn()
    if mode == "record":
        value = compute_fn()
        store.record(kind, key, value)
        return value
    return store.lookup(kind, key)            # replay: served from the GPU record; no forward
