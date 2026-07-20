"""C33 local joint-good boundary geometry."""
from __future__ import annotations

import numpy as np

from . import schema
from .artifact_loader import units


def _runs(labels):
    if not labels:
        return []
    out = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            out.append((labels[start], start, i - 1, i - start))
            start = i
    out.append((labels[start], start, len(labels) - 1, len(labels) - start))
    return out


def _autocorr(labels, lag):
    if len(labels) <= lag:
        return None
    a = np.array(labels[:-lag], dtype=float)
    b = np.array(labels[lag:], dtype=float)
    if a.std() <= 1e-9 or b.std() <= 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _transition_distance(selected_idx, labels):
    trans = [i for i in range(len(labels) - 1) if labels[i] != labels[i + 1]]
    if not trans:
        return None
    return float(min(min(abs(selected_idx - i), abs(selected_idx - (i + 1))) for i in trans))


def _neighborhood_rate(cs, selected_idx, k):
    lo = max(0, selected_idx - k)
    hi = min(len(cs), selected_idx + k + 1)
    vals = [c["joint_good"] for c in cs[lo:hi]]
    return float(np.mean(vals)) if vals else None


def boundary_geometry(rows):
    registry, autocorr_rows, neighborhood_rows = [], [], []
    for key, cs in units(rows).items():
        labels = [int(c["joint_good"]) for c in cs]
        rs = _runs(labels)
        good_runs = [r[3] for r in rs if r[0] == 1]
        trans = sum(1 for i in range(len(labels) - 1) if labels[i] != labels[i + 1])
        selected = [i for i, c in enumerate(cs) if c.get("selected_oaci")]
        selected_idx = selected[0] if len(selected) == 1 else None
        selected_boundary_distance = _transition_distance(selected_idx, labels) if selected_idx is not None else None
        registry.append({
            "seed": key[0], "target": key[1], "level": key[2], "regime": key[3], "n": len(cs),
            "joint_good_rate": float(np.mean(labels)) if labels else None,
            "n_good_runs": len(good_runs),
            "mean_good_run_length": float(np.mean(good_runs)) if good_runs else 0.0,
            "median_good_run_length": float(np.median(good_runs)) if good_runs else 0.0,
            "transition_count": trans,
            "transition_rate": trans / max(len(cs) - 1, 1),
            "selected_boundary_distance": selected_boundary_distance,
            "selected_joint_good": int(cs[selected_idx]["joint_good"]) if selected_idx is not None else None,
        })
        for lag in schema.AUTOCORR_LAGS:
            autocorr_rows.append({"seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
                                  "lag": lag, "autocorr": _autocorr(labels, lag)})
        if selected_idx is not None:
            for k in schema.ORDER_NEIGHBORHOODS:
                neighborhood_rows.append({"seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
                                          "neighborhood": f"pm{k}", "joint_good_rate": _neighborhood_rate(cs, selected_idx, k),
                                          "contains_joint_good": int((_neighborhood_rate(cs, selected_idx, k) or 0) > 0)})
    summary = {
        "n_units": len(registry),
        "mean_transition_rate": float(np.mean([r["transition_rate"] for r in registry])) if registry else None,
        "median_selected_boundary_distance": float(np.median([r["selected_boundary_distance"] for r in registry
                                                              if r["selected_boundary_distance"] is not None])) if registry else None,
        "mean_pm1_joint_good_rate": float(np.mean([r["joint_good_rate"] for r in neighborhood_rows
                                                   if r["neighborhood"] == "pm1"])) if neighborhood_rows else None,
        "pm1_contains_joint_fraction": float(np.mean([r["contains_joint_good"] for r in neighborhood_rows
                                                      if r["neighborhood"] == "pm1"])) if neighborhood_rows else None,
        "mean_good_run_length": float(np.mean([r["mean_good_run_length"] for r in registry])) if registry else None,
    }
    return {"summary": summary, "registry": registry, "autocorr": autocorr_rows, "neighborhoods": neighborhood_rows}
