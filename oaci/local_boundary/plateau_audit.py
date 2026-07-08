"""C33 source-score plateau / tie audit."""
from __future__ import annotations

import numpy as np

from . import schema
from .artifact_loader import units


def plateau_audit(rows):
    out = []
    for key, cs in units(rows).items():
        selected = [i for i, c in enumerate(cs) if c.get("selected_oaci")]
        if len(selected) != 1:
            continue
        si = selected[0]
        s = cs[si]
        plateau = [c for c in cs if abs(float(c["score"]) - float(s["score"])) <= schema.PLATEAU_EPS]
        sorted_plateau = sorted(plateau, key=lambda c: c["score"], reverse=True)
        selected_rank = next((i + 1 for i, c in enumerate(sorted_plateau) if c is s), None)
        out.append({
            "seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
            "plateau_size": len(plateau),
            "plateau_fraction": len(plateau) / len(cs) if cs else None,
            "plateau_joint_good_rate": float(np.mean([c["joint_good"] for c in plateau])) if plateau else None,
            "plateau_contains_joint_good": int(any(c["joint_good"] for c in plateau)),
            "selected_joint_good": int(s["joint_good"]),
            "selected_rank_within_plateau": selected_rank,
            "local_score_range_pm3": _local_range(cs, si, 3),
        })
    summary = {
        "n_units": len(out),
        "mean_plateau_size": float(np.mean([r["plateau_size"] for r in out])) if out else None,
        "median_plateau_size": float(np.median([r["plateau_size"] for r in out])) if out else None,
        "plateau_contains_joint_fraction": float(np.mean([r["plateau_contains_joint_good"] for r in out])) if out else None,
        "selected_bad_plateau_has_joint_fraction": float(np.mean([r["plateau_contains_joint_good"] for r in out
                                                                  if not r["selected_joint_good"]])) if out else None,
        "mean_plateau_joint_good_rate": float(np.mean([r["plateau_joint_good_rate"] for r in out])) if out else None,
    }
    return {"summary": summary, "plateaus": out}


def _local_range(cs, si, k):
    lo = max(0, si - k)
    hi = min(len(cs), si + k + 1)
    vals = [float(c["score"]) for c in cs[lo:hi]]
    return float(max(vals) - min(vals)) if vals else None
