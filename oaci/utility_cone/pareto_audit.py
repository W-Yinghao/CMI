"""Pareto and epsilon-Pareto audits for selected-vs-alternative pairs."""
from __future__ import annotations

import numpy as np

from . import endpoint_vectors, schema


def _status(delta):
    if np.all(delta > schema.UTILITY_WIN_EPS):
        return "strict_pareto_better"
    if np.all(delta >= -schema.UTILITY_WIN_EPS) and np.any(delta > schema.UTILITY_WIN_EPS):
        return "weak_pareto_better"
    if np.all(delta <= schema.UTILITY_WIN_EPS) and np.any(delta < -schema.UTILITY_WIN_EPS):
        return "selected_dominates_alternative"
    return "pareto_incomparable"


def pareto_audit(vector_rows):
    rows = []
    for r in vector_rows:
        delta = endpoint_vectors.vector_for(r, "raw")
        row = {
            "pair_id": r["pair_id"], "seed": r["seed"], "target": r["target"], "level": r["level"],
            "regime": r["regime"], "comparison": r["comparison"],
            "raw_delta_bacc": r["raw_delta_bacc"],
            "raw_delta_nll_improve": r["raw_delta_nll_improve"],
            "raw_delta_ece_improve": r["raw_delta_ece_improve"],
            "pareto_status": _status(delta),
            "endpoint_tradeoff": int(np.any(delta > schema.UTILITY_WIN_EPS) and np.any(delta < -schema.UTILITY_WIN_EPS)),
        }
        for eps in schema.EPSILON_PARETO:
            key = f"epsilon_{str(eps).replace('.', 'p')}_pareto_better"
            row[key] = int(np.all(delta >= -eps) and np.any(delta > eps))
        rows.append(row)
    summary = _summary(rows, "nearest_continuous_better")
    return {"summary": summary, "rows": rows}


def _summary(rows, comparison):
    rs = [r for r in rows if r["comparison"] == comparison]
    n = len(rs)
    counts = {}
    for r in rs:
        counts[r["pareto_status"]] = counts.get(r["pareto_status"], 0) + 1
    frac = lambda name: counts.get(name, 0) / n if n else None
    eps = {}
    for e in schema.EPSILON_PARETO:
        key = f"epsilon_{str(e).replace('.', 'p')}_pareto_better"
        eps[key + "_fraction"] = float(np.mean([r[key] for r in rs])) if rs else None
    return {
        "comparison": comparison,
        "n_pairs": n,
        "strict_pareto_better_fraction": frac("strict_pareto_better"),
        "weak_pareto_better_fraction": frac("weak_pareto_better"),
        "pareto_incomparable_fraction": frac("pareto_incomparable"),
        "selected_dominates_alternative_fraction": frac("selected_dominates_alternative"),
        "endpoint_tradeoff_fraction": float(np.mean([r["endpoint_tradeoff"] for r in rs])) if rs else None,
        "status_counts": counts,
        **eps,
    }
