"""Endpoint utility simplex audit for C35."""
from __future__ import annotations

import numpy as np

from . import endpoint_vectors, schema


def weight_grid(step=None):
    step = schema.UTILITY_GRID_STEP if step is None else step
    n = int(round(1.0 / step))
    weights = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            weights.append((i / n, j / n, k / n))
    return np.array(weights, dtype=float)


def cone_category(fraction):
    if fraction >= schema.ROBUST_WEIGHT_FRACTION:
        return "preference_robust_regret"
    if fraction >= schema.NARROW_WEIGHT_FRACTION:
        return "preference_dependent_regret"
    if fraction > 0:
        return "narrow_scalarization_regret"
    return "no_regret"


def simplex_for_vectors(vector_rows, scaling="raw"):
    weights = weight_grid()
    rows = []
    for r in vector_rows:
        delta = endpoint_vectors.vector_for(r, scaling)
        vals = weights @ delta
        wins = vals > schema.UTILITY_WIN_EPS
        best = weights[int(np.argmax(vals))]
        worst = weights[int(np.argmin(vals))]
        frac = float(np.mean(wins))
        rows.append({
            "pair_id": r["pair_id"], "seed": r["seed"], "target": r["target"], "level": r["level"],
            "regime": r["regime"], "comparison": r["comparison"], "scaling": scaling,
            "weight_grid_step": schema.UTILITY_GRID_STEP, "n_weights": len(weights),
            "fraction_weights_alt_beats_selected": frac,
            "utility_cone_category": cone_category(frac),
            "mean_regret_over_simplex": float(np.mean(vals)),
            "min_regret_over_simplex": float(np.min(vals)),
            "max_regret_over_simplex": float(np.max(vals)),
            "best_weight_bacc": float(best[0]), "best_weight_nll": float(best[1]),
            "best_weight_ece": float(best[2]),
            "worst_weight_bacc": float(worst[0]), "worst_weight_nll": float(worst[1]),
            "worst_weight_ece": float(worst[2]),
        })
    return rows


def simplex_audit(vector_rows):
    rows = simplex_for_vectors(vector_rows, "raw")
    summary = cone_summary(rows, "nearest_continuous_better")
    return {"summary": summary, "rows": rows}


def cone_summary(rows, comparison="nearest_continuous_better"):
    rs = [r for r in rows if r["comparison"] == comparison]
    n = len(rs)
    counts = {}
    for r in rs:
        counts[r["utility_cone_category"]] = counts.get(r["utility_cone_category"], 0) + 1
    frac = lambda name: counts.get(name, 0) / n if n else None
    vals = [r["fraction_weights_alt_beats_selected"] for r in rs]
    return {
        "comparison": comparison,
        "n_pairs": n,
        "preference_robust_fraction": frac("preference_robust_regret"),
        "preference_dependent_fraction": frac("preference_dependent_regret"),
        "narrow_scalarization_fraction": frac("narrow_scalarization_regret"),
        "no_regret_fraction": frac("no_regret"),
        "mean_weight_fraction_alt_wins": float(np.mean(vals)) if vals else None,
        "median_weight_fraction_alt_wins": float(np.median(vals)) if vals else None,
        "category_counts": counts,
    }
