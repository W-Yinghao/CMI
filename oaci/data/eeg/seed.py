"""Offline SEED scanner (feasibility / header). The full window->film-clip epoching loader is
wired in the next commit; here we report availability + structure so the data pre-check can
validate the path and the eval-unit definition (window aggregated to film clip).
"""
from __future__ import annotations

import os

from .registry import DATALAKE


def scan_seed(root: str | None = None) -> dict:
    root = root or os.path.join(DATALAKE, "SEED-V")
    if not os.path.exists(root):
        return {"available": False, "path": root, "reason": "offline path not found (not downloaded)"}
    try:
        top = sorted(os.listdir(root))
    except OSError as e:
        return {"available": False, "path": root, "reason": str(e)}
    return {
        "available": True, "path": root, "top_level": top[:12],
        "eval_unit": "film clip (windows aggregated by mean probability)",
        "loader_status": "scanner-only (full epoching deferred to the EEG-trainer commit)",
    }
