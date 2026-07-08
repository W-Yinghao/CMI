"""Leakage-UCL direction vs target gauge / representation-projection geometry."""
from __future__ import annotations

from statistics import mean

from . import artifact_loader as al
from . import schema


def audit(ctx, ucl_rows):
    rows = []
    for ur in ucl_rows:
        c34 = al.c34_for_exact({"pair_id": ur["pair_id"]}, ctx)
        gauge_delta = al.as_float(c34.get("target_gauge_delta"))
        margin_delta = al.as_float(c34.get("target_margin_mean_delta"))
        gauge_pref = al.pref_from_delta(
            gauge_delta, schema.GAUGE_CLEAR_EPS, positive_prefers="better", negative_prefers="selected")
        margin_pref = al.pref_from_delta(
            margin_delta, schema.GAUGE_CLEAR_EPS, positive_prefers="better", negative_prefers="selected")
        conflict = int(ur["ucl_prefers"] == "selected" and gauge_pref == "better")
        rows.append({
            "pair_id": ur["pair_id"],
            "pair_key": ur["pair_key"],
            "seed": ur["seed"],
            "target": ur["target"],
            "level": ur["level"],
            "regime": ur["regime"],
            "selection_ucl_prefers": ur["ucl_prefers"],
            "target_endpoint_prefers": ur["target_endpoint_prefers"],
            "target_gauge_delta_better_minus_selected": gauge_delta,
            "target_gauge_prefers": gauge_pref,
            "target_margin_mean_delta_better_minus_selected": margin_delta,
            "target_margin_mean_prefers": margin_pref,
            "leakage_target_gauge_conflict": conflict,
            "target_gauge_non_source_only": 1,
            "c27_class_conditioned_confidence_global_available": 1,
            "c29_representation_projection_global_available": 1,
            "pair_local_representation_projection_atom_available": 0,
        })
    n = len(rows)
    summary = {
        "n_pairs": n,
        "target_gauge_prefers_better_count": sum(1 for r in rows if r["target_gauge_prefers"] == "better"),
        "target_gauge_prefers_selected_count": sum(1 for r in rows if r["target_gauge_prefers"] == "selected"),
        "leakage_target_gauge_conflict_count": sum(r["leakage_target_gauge_conflict"] for r in rows),
        "leakage_target_gauge_conflict_fraction": mean(
            [r["leakage_target_gauge_conflict"] for r in rows]) if rows else None,
        "mean_target_gauge_delta": mean(
            [r["target_gauge_delta_better_minus_selected"] for r in rows]) if rows else None,
        "pair_local_representation_projection_atom_available": False,
    }
    return {"rows": rows, "summary": summary}

