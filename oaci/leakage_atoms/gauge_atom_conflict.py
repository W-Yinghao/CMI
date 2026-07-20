"""Dominant leakage atom direction vs target-gauge diagnostics."""
from __future__ import annotations

from collections import defaultdict

from . import artifact_loader as al
from . import schema


def audit(ctx, point_atom_rows):
    by_pair = defaultdict(list)
    for r in point_atom_rows:
        by_pair[r["pair_id"]].append(r)
    rows = []
    for pair_id, rs in sorted(by_pair.items()):
        top = max(rs, key=lambda r: float(r["positive_selected_advantage"]))
        gauge = ctx["by_pair"]["c38_gauge"][pair_id]
        endpoint = ctx["by_pair"]["c35_endpoint"].get(pair_id, {})
        gauge_delta = al.as_float(gauge["target_gauge_delta_better_minus_selected"])
        gauge_pref = al.pref_from_delta(
            gauge_delta, schema.GAUGE_CLEAR_EPS, positive_prefers="better", negative_prefers="selected")
        target_pref = gauge["target_endpoint_prefers"]
        conflict = int(top["selected_advantage_sign"] == "selected" and gauge_pref == "better")
        rows.append({
            "pair_id": pair_id,
            "pair_key": top["pair_key"],
            "seed": top["seed"],
            "target": top["target"],
            "level": top["level"],
            "regime": top["regime"],
            "selected_order": top["selected_order"],
            "better_order": top["better_order"],
            "dominant_atom_id": top["atom_id"],
            "dominant_class_id": top["class_id"],
            "dominant_class_name": top["class_name"],
            "dominant_domain_id": top["domain_id"],
            "dominant_domain_name": top["domain_name"],
            "dominant_atom_delta_better_minus_selected": top["atom_delta_better_minus_selected"],
            "dominant_atom_positive_share": top["positive_advantage_share"],
            "dominant_atom_prefers": top["selected_advantage_sign"],
            "target_gauge_delta_better_minus_selected": gauge_delta,
            "target_gauge_prefers": gauge_pref,
            "target_endpoint_prefers": target_pref,
            "target_bacc_delta": endpoint.get("raw_delta_bacc", ""),
            "target_nll_improve_delta": endpoint.get("raw_delta_nll_improve", ""),
            "target_ece_improve_delta": endpoint.get("raw_delta_ece_improve", ""),
            "atom_target_gauge_conflict": conflict,
            "target_gauge_non_source_only": 1,
        })
    summary = {
        "n_pairs": len(rows),
        "target_gauge_prefers_better_count": sum(1 for r in rows if r["target_gauge_prefers"] == "better"),
        "target_gauge_prefers_selected_count": sum(1 for r in rows if r["target_gauge_prefers"] == "selected"),
        "atom_target_gauge_conflict_count": sum(int(r["atom_target_gauge_conflict"]) for r in rows),
        "atom_target_gauge_conflict_fraction": (
            sum(int(r["atom_target_gauge_conflict"]) for r in rows) / len(rows) if rows else None),
        "mean_dominant_atom_positive_share": al.finite_mean(
            [r["dominant_atom_positive_share"] for r in rows]),
    }
    return {"rows": rows, "summary": summary}
