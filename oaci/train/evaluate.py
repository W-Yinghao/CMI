"""Chunked guard evaluator — additive sufficient statistics, never an average of chunk means.

``model.eval()`` (per-submodule restore) + ``torch.inference_mode()`` so no buffer or RNG is
touched. ``balanced_ce``/``balanced_err`` accumulate per-class weighted numerator and mass across
chunks, then average once — numerically identical for any (uneven) partition. A pre-registered
class with zero mass is a LOUD failure (never a silent class drop).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .bn import all_eval

_METRICS = ("ce", "balanced_ce")


@dataclass(frozen=True)
class GuardResult:
    risk: float
    balanced_err: float
    metric: str


def evaluate_guard(model, data, metric: str, chunk_size=None, device=None) -> GuardResult:
    if metric not in _METRICS:
        raise ValueError(f"guard metric must be one of {_METRICS}; got {metric!r}")
    nc = int(data.n_classes)
    X, y, w = data.X, data.y, data.sample_mass
    n = int(X.shape[0])
    cs = n if chunk_size is None else int(chunk_size)
    if cs <= 0:
        raise ValueError("chunk_size must be positive")
    dev = device or X.device

    # accumulate on the compute device (== cpu for CPU runs, so CPU results are byte-identical); on GPU
    # this keeps the additive sufficient statistics on-device instead of mixing cuda operands with cpu scalars.
    ce_num = torch.zeros((), dtype=torch.float64, device=dev)   # for global ce
    ce_den = torch.zeros((), dtype=torch.float64, device=dev)
    pc_num = torch.zeros(nc, dtype=torch.float64, device=dev)   # per-class CE numerator/mass
    pc_den = torch.zeros(nc, dtype=torch.float64, device=dev)
    err_num = torch.zeros(nc, dtype=torch.float64, device=dev)  # per-class weighted errors/mass

    with all_eval(model), torch.inference_mode():
        for a in range(0, n, cs):
            xb = X[a:a + cs].to(dev)
            yb = y[a:a + cs].to(dev)
            wb = w[a:a + cs].to(device=dev, dtype=torch.float32)
            logits = model(xb).logits
            if logits.dim() != 2 or logits.shape[1] != nc:
                raise ValueError(f"guard logits shape {tuple(logits.shape)} != (B,{nc})")
            if not torch.isfinite(logits).all():
                raise ValueError("guard logits are not finite")
            per = F.cross_entropy(logits, yb, reduction="none").double()
            wd = wb.double()
            wrong = (logits.argmax(1) != yb).double()
            ce_num += (wd * per).sum(); ce_den += wd.sum()
            for c in range(nc):
                m = yb == c
                if bool(m.any()):
                    pc_num[c] += (wd[m] * per[m]).sum()
                    pc_den[c] += wd[m].sum()
                    err_num[c] += (wd[m] * wrong[m]).sum()

    if metric == "ce":
        risk = float(ce_num / ce_den.clamp_min(1e-300))
    else:
        if bool((pc_den <= 0).any()):
            missing = [c for c in range(nc) if float(pc_den[c]) <= 0]
            raise ValueError(f"balanced guard: pre-registered class(es) {missing} have zero mass")
        risk = float((pc_num / pc_den).mean())
    if bool((pc_den <= 0).any()):
        # balanced_err is a report metric; require all classes present for it too (consistent guard)
        missing = [c for c in range(nc) if float(pc_den[c]) <= 0]
        raise ValueError(f"balanced_err: pre-registered class(es) {missing} have zero mass")
    bal_err = float((err_num / pc_den).mean())
    return GuardResult(risk=risk, balanced_err=bal_err, metric=metric)
