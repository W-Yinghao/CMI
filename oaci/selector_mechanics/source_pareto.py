"""C36 source-side Pareto conflict audit."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import schema


def _vec(selected, better):
    vals = []
    for obj in schema.SOURCE_PARETO_OBJECTIVES:
        s = al.as_float(selected.get(obj["field"]))
        b = al.as_float(better.get(obj["field"]))
        vals.append(obj["orientation"] * (b - s))
    return vals


def _status(vals):
    finite_vals = [v for v in vals if al.finite(v)]
    if not finite_vals:
        return "source_pareto_unavailable"
    eps = schema.SOURCE_PARETO_EPS
    if all(v >= -eps for v in finite_vals) and any(v > eps for v in finite_vals):
        return "better_source_dominates_selected"
    if all(v <= eps for v in finite_vals) and any(v < -eps for v in finite_vals):
        return "selected_source_dominates_better"
    if all(abs(v) <= eps for v in finite_vals):
        return "source_pareto_tie"
    return "source_pareto_incomparable"


def audit(pair_rows, trace):
    rows = []
    rational = []
    for r in pair_rows:
        s = trace["by_key"][(r["seed"], r["target"], r["level"], r["regime"], str(r["selected_order"]))]
        b = trace["by_key"][(r["seed"], r["target"], r["level"], r["regime"], str(r["better_order"]))]
        vals = _vec(s, b)
        status = _status(vals)
        out = {
            "pair_id": r["pair_id"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "source_pareto_status": status,
            "target_prefers": "better",
            "n_source_objectives_finite": sum(al.finite(v) for v in vals),
            "n_source_objectives_registered": len(schema.SOURCE_PARETO_OBJECTIVES),
        }
        for obj, val in zip(schema.SOURCE_PARETO_OBJECTIVES, vals):
            out[f"{obj['objective']}_oriented_delta_better_minus_selected"] = val
        rows.append(out)
        if status in ("selected_source_dominates_better", "source_pareto_incomparable"):
            rational.append({
                "pair_id": r["pair_id"],
                "seed": r["seed"],
                "target": r["target"],
                "level": r["level"],
                "regime": r["regime"],
                "source_pareto_status": status,
                "target_prefers": "better",
                "source_rational_not_better_dominated": 1,
            })
    n = len(rows)
    counts = {k: sum(1 for r in rows if r["source_pareto_status"] == k)
              for k in ("better_source_dominates_selected", "selected_source_dominates_better",
                        "source_pareto_incomparable", "source_pareto_tie", "source_pareto_unavailable")}
    summary = {f"{k}_count": v for k, v in counts.items()}
    summary.update({f"{k}_fraction": (v / n if n else None) for k, v in counts.items()})
    summary["n_pairs"] = n
    summary["source_pareto_conflict_fraction"] = (
        (counts["selected_source_dominates_better"] + counts["source_pareto_incomparable"]) / n if n else None)
    summary["better_source_dominates_fraction"] = counts["better_source_dominates_selected"] / n if n else None
    return {"rows": rows, "summary": summary, "source_rational_rows": rational,
            "objective_registry": list(schema.SOURCE_PARETO_OBJECTIVES)}
