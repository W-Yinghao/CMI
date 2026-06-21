"""Phase-2 estimand. THE ONLY MODULE THAT READS y_target. Computes paired incremental risk
    ΔR_a(B) = R_B(f_a) − R_B(f_0)
on the WHOLE batch (no label-conditioned subsetting — that was the A0' leak). Positive ΔR_a = negative transfer.

Kept strictly separate from features.py so the metamorphic guard can prove the scoring path never reaches here.
"""
from __future__ import annotations
import numpy as np


def _nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)).mean())


def _bal_err(p, y):
    pred = p.argmax(1)
    errs = []
    for c in np.unique(y):
        m = y == c
        if m.any():
            errs.append(float((pred[m] != c).mean()))
    return float(np.mean(errs)) if errs else float((pred != y).mean())


def batch_risk(p, y, kind="nll"):
    return _nll(p, y) if kind == "nll" else _bal_err(p, y)


def delta_risk(p0, pa, y, kind="nll"):
    """ΔR_a(B) — whole batch. >0 ⇒ action harmed; <0 ⇒ action helped."""
    return batch_risk(pa, y, kind) - batch_risk(p0, y, kind)


def harm_label(dr):
    """Binary negative-transfer label for AUROC. 1 = action made this batch worse."""
    return int(dr > 0)
