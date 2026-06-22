"""SHA-256 derived, namespace-separated RNG — never Python ``hash()`` (salted, unstable).

Each stochastic forward runs inside ``forked_rng`` so a critic call cannot perturb the task
dropout stream, method execution order cannot leak across methods, and the same base seed
reproduces every checkpoint hash.
"""
from __future__ import annotations

import contextlib
import hashlib

import torch

_MASK = (1 << 31) - 1


def derive_seed(base_seed: int, namespace: str, *parts) -> int:
    """Deterministic 31-bit seed from ``(base_seed, namespace, *parts)`` via SHA-256. Stable across
    processes (unlike ``hash()``) and separated by namespace, so distinct streams never collide."""
    h = hashlib.sha256()
    h.update(int(base_seed).to_bytes(8, "little", signed=True))
    h.update(b"\x00"); h.update(str(namespace).encode())
    for p in parts:
        h.update(b"\x00"); h.update(str(p).encode())
    return int.from_bytes(h.digest()[:8], "little") & _MASK


@contextlib.contextmanager
def forked_rng(seed: int, device=None):
    """Run with a locally-set CPU (and CUDA, if applicable) RNG, restoring the caller's RNG state
    on exit — so engine randomness is decided by the derived ``seed``, not the ambient global RNG,
    and the caller's stream is left untouched."""
    use_cuda = device is not None and getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available()
    devices = [device] if use_cuda else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if use_cuda:
            torch.cuda.manual_seed_all(int(seed))
        yield
