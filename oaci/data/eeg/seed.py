"""Offline scanner for the CANONICAL 3-class SEED (15 subjects x 3 sessions x 15 film clips).

SEED-V (5-emotion) is a DIFFERENT dataset and is NOT a substitute — the registry declares the
3-class SEED, so the scanner looks ONLY for canonical SEED and reports ``unavailable`` if absent
(never silently loads SEED-V). The full window->film-clip epoching loader is the next commit.
"""
from __future__ import annotations

import os

from .registry import DATALAKE


def canonical_seed_path() -> str:
    return os.path.join(DATALAKE, "SEED")            # canonical SEED, NOT SEED-V


def scan_seed(root: str | None = None) -> dict:
    root = root or canonical_seed_path()
    if "SEED-V" in os.path.basename(os.path.normpath(root)):
        return {"available": False, "path": root, "is_canonical_seed": False,
                "reason": "SEED-V is a different (5-emotion) dataset, NOT a substitute for 3-class SEED"}
    if not os.path.exists(root):
        return {"available": False, "path": root, "is_canonical_seed": True,
                "reason": "canonical 3-class SEED not found offline (SEED-V is NOT used as a substitute)"}
    return {"available": True, "path": root, "is_canonical_seed": True,
            "n_subjects_expected": 15, "n_sessions_expected": 3, "n_clips_expected": 15,
            "loader_status": "scanner-only (window->film-clip epoching deferred to the EEG-trainer commit)"}
