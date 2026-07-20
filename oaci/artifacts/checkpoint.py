"""Content-addressed checkpoint store.

A checkpoint is a ``dict[str, torch.Tensor]`` (CPU, dense, contiguous). The scientific identity is the
core ``state_hash`` (key + dtype + shape + bytes); ``file_sha256`` is only a transport/corruption
check. The same model hash is written once; a second write of the same hash with different bytes is a
hard failure. Loads use ``weights_only=True`` and re-verify the state hash.
"""
from __future__ import annotations

import hashlib
import os

import torch

from ..train.checkpoint import state_hash

CHECKPOINT_WRITER_VERSION = "oaci-ckpt-v1"


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_state(state) -> dict:
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a dict")
    out = {}
    for k, v in state.items():
        if not isinstance(k, str):
            raise TypeError(f"checkpoint key must be str, got {type(k)!r}")
        if not torch.is_tensor(v):
            raise TypeError(f"checkpoint value for {k!r} must be a Tensor, got {type(v)!r}")
        out[k] = v.detach().cpu().contiguous()
    return out


def checkpoint_metadata(model_hash, state, file_sha256) -> dict:
    return {"model_hash": model_hash, "writer_version": CHECKPOINT_WRITER_VERSION, "file_sha256": file_sha256,
            "tensors": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in state.items()}}


def write_checkpoint_file(pt_path, model_hash, state) -> dict:
    clean = _clean_state(state)
    if state_hash(clean) != model_hash:
        raise ValueError(f"checkpoint state hash != model hash {model_hash}")
    with open(pt_path, "wb") as f:
        torch.save(clean, f)
        f.flush(); os.fsync(f.fileno())
    fsha = _file_sha256(pt_path)
    reloaded = torch.load(pt_path, map_location="cpu", weights_only=True)   # round-trip immediately
    rc = _clean_state(reloaded)
    if state_hash(rc) != model_hash:
        raise RuntimeError("checkpoint state hash changed across a write/read round-trip")
    return checkpoint_metadata(model_hash, clean, fsha)


def read_checkpoint_file(pt_path, metadata) -> dict:
    if _file_sha256(pt_path) != metadata["file_sha256"]:
        raise ValueError(f"checkpoint file sha256 mismatch (corruption): {pt_path}")
    state = _clean_state(torch.load(pt_path, map_location="cpu", weights_only=True))
    if state_hash(state) != metadata["model_hash"]:
        raise ValueError(f"checkpoint state hash mismatch on read: {pt_path}")
    for k, t in state.items():
        m = metadata["tensors"].get(k)
        if m is None or str(t.dtype) != m["dtype"] or list(t.shape) != list(m["shape"]):
            raise ValueError(f"checkpoint tensor {k!r} dtype/shape disagrees with metadata")
    return state


class CheckpointStore:
    """Accumulate unique checkpoints for a level; physical file written once per model hash."""

    def __init__(self):
        self._states = {}

    def add(self, model_hash, state) -> None:
        clean = _clean_state(state)
        if state_hash(clean) != model_hash:
            raise ValueError(f"checkpoint state hash != model hash {model_hash}")
        prev = self._states.get(model_hash)
        if prev is not None and state_hash(prev) != state_hash(clean):
            raise ValueError(f"two different tensor states share model hash {model_hash}")
        self._states[model_hash] = clean

    def model_hashes(self):
        return tuple(sorted(self._states))

    def state(self, model_hash):
        return self._states[model_hash]
