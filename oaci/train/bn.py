"""Per-submodule training-mode context managers + BatchNorm-buffer guards.

``model.train(bool)`` alone is insufficient for Stage-2: the parent must be in TRAIN (dropout
active) while every BatchNorm submodule is in EVAL (running stats frozen, used not updated). These
helpers save/restore EACH submodule's ``.training`` flag and verify BN buffers never move.
"""
from __future__ import annotations

import contextlib
import hashlib

import torch
import torch.nn as nn

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


@contextlib.contextmanager
def submodule_modes_restored(model: nn.Module):
    """Restore EVERY submodule's ``.training`` flag on exit (not just the root)."""
    saved = [(m, m.training) for m in model.modules()]
    try:
        yield
    finally:
        for m, t in saved:
            m.training = t


@contextlib.contextmanager
def all_eval(model: nn.Module):
    """Put the whole model in eval for the duration, restoring per-submodule modes after."""
    with submodule_modes_restored(model):
        model.eval()
        yield


def bn_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, _BN)]


def freeze_bn_running_stats(model: nn.Module) -> None:
    """Stage-2 mode: parent stays TRAIN, but every BatchNorm goes to EVAL so it USES (never
    updates) the ERM running stats. Affine ``weight``/``bias`` ``requires_grad`` is left untouched
    (they remain trainable)."""
    for m in bn_modules(model):
        m.eval()


def bn_buffer_hash(model: nn.Module) -> str:
    """Hash of all BatchNorm running buffers (running_mean/var/num_batches_tracked)."""
    h = hashlib.sha256()
    for i, m in enumerate(bn_modules(model)):
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            b = getattr(m, name, None)
            if b is None:
                continue
            t = b.detach().cpu().contiguous()
            h.update(f"{i}.{name}.{t.dtype}".encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()
