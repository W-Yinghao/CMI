"""Unified RNG-state snapshots (python random + numpy + torch CPU + one CUDA device).

A snapshot is taken at each scientific boundary (model factory, Stage-1, each Stage-2 method, feature /
audit extraction, prediction, the whole two-level run) and compared before/after to prove the engine
never perturbs the ambient global RNG.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RNGStateSnapshot:
    python_random_hash: str
    numpy_global_hash: str
    torch_cpu_hash: str
    torch_cuda_hash: str | None
    snapshot_hash: str


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def snapshot_rng_state(device=None) -> RNGStateSnapshot:
    py = _h(repr(random.getstate()).encode())
    npst = np.random.get_state()
    np_h = _h(npst[1].tobytes() + str((npst[0], int(npst[2]), int(npst[3]), float(npst[4]))).encode())
    cpu = _h(torch.random.get_rng_state().numpy().tobytes())
    cuda = None
    if device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available():
        idx = torch.device(device).index
        idx = 0 if idx is None else int(idx)
        cuda = _h(torch.cuda.get_rng_state(idx).numpy().tobytes())
    snap = _h("|".join([py, np_h, cpu, cuda or "-"]).encode())
    return RNGStateSnapshot(py, np_h, cpu, cuda, snap)


def assert_rng_unchanged(before: RNGStateSnapshot, after: RNGStateSnapshot, where: str) -> None:
    if before.snapshot_hash != after.snapshot_hash:
        diff = [n for n, a, b in (("python", before.python_random_hash, after.python_random_hash),
                                  ("numpy", before.numpy_global_hash, after.numpy_global_hash),
                                  ("torch_cpu", before.torch_cpu_hash, after.torch_cpu_hash),
                                  ("torch_cuda", before.torch_cuda_hash, after.torch_cuda_hash)) if a != b]
        raise RuntimeError(f"{where}: ambient RNG state changed ({', '.join(diff)})")
