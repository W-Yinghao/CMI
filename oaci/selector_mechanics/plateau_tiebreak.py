"""C36 selector plateau/tie diagnostics."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def _point_plateau(unit_rows, selected_point):
    out = []
    for r in unit_rows:
        if r["is_erm"]:
            continue
        v = al.as_float(r.get("selection_leakage_point"))
        if al.finite(v) and abs(v - selected_point) <= schema.POINT_PLATEAU_EPS:
            out.append(r)
    return out


def audit(pair_rows, trace):
    rows = []
    for r in pair_rows:
        key = (r["seed"], r["target"], r["level"], r["regime"])
        selected = trace["by_key"][(r["seed"], r["target"], r["level"], r["regime"], str(r["selected_order"]))]
        selected_point = al.as_float(selected.get("selection_leakage_point"))
        plateau = _point_plateau(trace["unit_rows"][key], selected_point) if al.finite(selected_point) else []
        better_inside = any(int(p["candidate_order"]) == int(r["better_order"]) for p in plateau)
        point_delta = al.as_float(r["selection_leakage_point_delta_better_minus_selected"])
        rows.append({
            "pair_id": r["pair_id"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "actual_selector_plateau_available": 0,
            "actual_selector_tie_break_metadata_available": 0,
            "actual_selector_tie_break_classification": "unavailable_no_per_candidate_ucl",
            "point_component_plateau_eps": schema.POINT_PLATEAU_EPS,
            "point_component_plateau_size": len(plateau),
            "better_in_point_component_plateau": int(better_inside),
            "point_component_active_selected_margin": int(al.finite(point_delta) and
                                                          point_delta > schema.POINT_PLATEAU_EPS),
        })
    n = len(rows)
    summary = {
        "n_pairs": n,
        "actual_selector_plateau_available": False,
        "actual_selector_tie_metadata_available": False,
        "point_component_plateau_fraction": (
            sum(r["better_in_point_component_plateau"] for r in rows) / n if n else None),
        "point_component_active_selected_margin_fraction": (
            sum(r["point_component_active_selected_margin"] for r in rows) / n if n else None),
    }
    return {"rows": rows, "summary": summary}

