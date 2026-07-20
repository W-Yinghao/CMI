"""One-way variance decomposition diagnostics for C46."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import schema


def _values(rows, outcome):
    vals = []
    for r in rows:
        v = al.as_float(r.get(outcome))
        if al.finite(v):
            vals.append(v)
    return vals


def _eta_squared(ctx, grouping, outcome):
    groups = al.group_rows(ctx, grouping)
    all_vals = np.asarray(_values(ctx["registry"], outcome), dtype=float)
    if len(all_vals) < 2:
        return None
    mu = float(np.mean(all_vals))
    total = float(np.sum((all_vals - mu) ** 2))
    if total <= 1e-12:
        return None
    between = 0.0
    within = 0.0
    used = 0
    for rows in groups.values():
        vals = np.asarray(_values(rows, outcome), dtype=float)
        if len(vals) == 0:
            continue
        gmu = float(np.mean(vals))
        between += len(vals) * (gmu - mu) ** 2
        within += float(np.sum((vals - gmu) ** 2))
        used += 1
    return {
        "outcome": outcome,
        "component": grouping,
        "n_groups": used,
        "eta_squared": between / total,
        "within_fraction": within / total,
        "total_variance": total / len(all_vals),
        "diagnostic_only": 1,
    }


def audit(ctx):
    rows = []
    for outcome in schema.OUTCOMES:
        for grouping in ("target", "trajectory", "seed", "level", "regime", "target_regime"):
            row = _eta_squared(ctx, grouping, outcome)
            if row is not None:
                rows.append(row)
        traj = next(r for r in rows if r["outcome"] == outcome and r["component"] == "trajectory")
        rows.append({
            "outcome": outcome,
            "component": "residual_within_trajectory",
            "n_groups": traj["n_groups"],
            "eta_squared": traj["within_fraction"],
            "within_fraction": traj["within_fraction"],
            "total_variance": traj["total_variance"],
            "diagnostic_only": 1,
        })
    return {"rows": rows, "summary": {(r["outcome"], r["component"]): r for r in rows}}
