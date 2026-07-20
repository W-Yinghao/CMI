"""Endpoint scaling sensitivity for utility-cone conclusions."""
from __future__ import annotations

from . import utility_simplex


SCALINGS = ("raw", "global_z", "within_z", "rank")


def scaling_sensitivity(vector_rows):
    rows = []
    per_scaling_pair_rows = {}
    for scaling in SCALINGS:
        pair_rows = utility_simplex.simplex_for_vectors(vector_rows, scaling)
        per_scaling_pair_rows[scaling] = pair_rows
        summary = utility_simplex.cone_summary(pair_rows)
        rows.append({"scaling": scaling, **summary})
    robust_vals = [r["preference_robust_fraction"] for r in rows if r["preference_robust_fraction"] is not None]
    dep_vals = [r["preference_dependent_fraction"] for r in rows if r["preference_dependent_fraction"] is not None]
    narrow_vals = [r["narrow_scalarization_fraction"] for r in rows if r["narrow_scalarization_fraction"] is not None]
    summary = {
        "robust_fraction_range": (max(robust_vals) - min(robust_vals)) if robust_vals else None,
        "dependent_fraction_range": (max(dep_vals) - min(dep_vals)) if dep_vals else None,
        "narrow_fraction_range": (max(narrow_vals) - min(narrow_vals)) if narrow_vals else None,
        "scalings": list(SCALINGS),
    }
    return {"summary": summary, "rows": rows, "pair_rows_by_scaling": per_scaling_pair_rows}
