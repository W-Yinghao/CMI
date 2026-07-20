"""Additive artifact support for the C7 K1/K2 decision subtree (``levels/<level>/decisions/``).

Decisions are written THROUGH the writer's index (``_Tree.json`` — so ``k1.json`` / ``k1.npz`` / ``k2.json``
are indexed and ``verify_artifact_tree`` verifies them like any other file). Old artifacts (none written)
verify legacy-complete because they simply have no such files; ``verify_decisions(require=True)`` is a
SEPARATE check that confirms the decisions are present + well-formed when the manifest requests them."""
from __future__ import annotations

import glob
import os

import numpy as np

from .canonical_json import canonical_json_hash
from .reader import read_doc

_LEVELS = "levels"


def decisions_prefix(level: int) -> str:
    return f"{_LEVELS}/level-{int(level):03d}/decisions"


def add_level_decisions(t, level: int, k1_body: dict, k1_null_arrays: dict, k2_body: dict) -> None:
    """Write ``k1.json`` (+ companion ``k1.npz`` holding the permutation null) and ``k2.json`` into the
    writer tree ``t`` — all INDEXED, so the committed tree stays whole under verification."""
    base = decisions_prefix(level)
    t.json(f"{base}/k1", "k1_decision", canonical_json_hash(k1_body), k1_body, k1_null_arrays)
    t.json(f"{base}/k2", "k2_decision", canonical_json_hash(k2_body), k2_body)


def _levels_on_disk(artifact_dir: str) -> list:
    return sorted(int(os.path.basename(p).rsplit("-", 1)[-1])
                  for p in glob.glob(os.path.join(artifact_dir, _LEVELS, "level-*")) if os.path.isdir(p))


def has_level_decisions(artifact_dir: str, level: int) -> bool:
    base = os.path.join(artifact_dir, decisions_prefix(level))
    return all(os.path.exists(os.path.join(base, f)) for f in ("k1.json", "k1.npz", "k2.json"))


def read_level_decisions(artifact_dir: str, level: int) -> dict:
    base = os.path.join(artifact_dir, decisions_prefix(level))
    k1 = read_doc(os.path.join(base, "k1.json")); k2 = read_doc(os.path.join(base, "k2.json"))
    npz = np.load(os.path.join(base, "k1.npz"))
    return {"k1": k1.get("body", k1), "k2": k2.get("body", k2), "k1_null": {k: npz[k] for k in npz.files}}


def verify_decisions(artifact_dir: str, *, require: bool) -> dict:
    """Legacy-tolerant when ``require`` is False; when True, every level must carry a well-formed decision."""
    levels = _levels_on_disk(artifact_dir)
    with_dec = [L for L in levels if has_level_decisions(artifact_dir, L)]
    if require:
        missing = [L for L in levels if L not in with_dec]
        if missing:
            raise ValueError(f"manifest requests K1/K2 decisions but levels {missing} lack them")
        for L in levels:
            rec = read_level_decisions(artifact_dir, L)
            if "k1_status" not in rec["k1"] or "k2_status" not in rec["k2"]:
                raise ValueError(f"level {L} decision payload malformed (missing k1_status/k2_status)")
            if "null" not in rec["k1_null"]:
                raise ValueError(f"level {L} k1 permutation null array missing")
    return {"levels": levels, "with_decisions": with_dec, "required": bool(require)}
