"""Selection UCL = point + bootstrap-width decomposition."""
from __future__ import annotations

from statistics import mean

from . import artifact_loader as al
from . import schema


def _classify(dp, dw, du):
    if not al.finite(du) or abs(float(du)) <= schema.UCL_CLEAR_EPS:
        return "flat_or_unavailable"
    if dp > schema.POINT_CLEAR_EPS and dw > schema.WIDTH_CLEAR_EPS:
        return "mixed_point_dominant" if abs(dp) >= abs(dw) else "mixed_width_dominant"
    if dp > schema.POINT_CLEAR_EPS and dw <= schema.WIDTH_CLEAR_EPS:
        return "point_driven_width_offsets"
    if dp <= schema.POINT_CLEAR_EPS and dw > schema.WIDTH_CLEAR_EPS:
        return "uncertainty_driven"
    return "opposed_or_unclassified"


def decompose(ctx):
    rows = []
    for r in ctx["tables"]["c37"]["exact"]:
        sp = al.as_float(r["selected_point"])
        bp = al.as_float(r["better_point"])
        su = al.as_float(r["selected_ucl"])
        bu = al.as_float(r["better_ucl"])
        sw = su - sp
        bw = bu - bp
        dp = bp - sp
        dw = bw - sw
        du = bu - su
        denom = abs(dp) + abs(dw)
        rows.append({
            "pair_id": r["pair_id"],
            "pair_key": r["pair_key"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "selected_order": r["selected_order"],
            "better_order": r["better_order"],
            "selected_point": sp,
            "better_point": bp,
            "point_delta_better_minus_selected": dp,
            "selected_width": sw,
            "better_width": bw,
            "width_delta_better_minus_selected": dw,
            "selected_ucl": su,
            "better_ucl": bu,
            "ucl_delta_better_minus_selected": du,
            "point_share_abs": abs(dp) / denom if denom else None,
            "width_share_abs": abs(dw) / denom if denom else None,
            "signed_point_fraction_of_ucl_margin": dp / du if al.finite(du) and abs(du) > 0 else None,
            "signed_width_fraction_of_ucl_margin": dw / du if al.finite(du) and abs(du) > 0 else None,
            "point_prefers": al.pref_from_delta(dp, schema.POINT_CLEAR_EPS),
            "width_prefers": al.pref_from_delta(dw, schema.WIDTH_CLEAR_EPS),
            "ucl_prefers": al.pref_from_delta(du, schema.UCL_CLEAR_EPS),
            "target_endpoint_prefers": r["target_endpoint_prefers"],
            "point_width_class": _classify(dp, dw, du),
        })
    n = len(rows)
    count = lambda key, val: sum(1 for r in rows if r[key] == val)
    point_dom = sum(1 for r in rows if r["point_prefers"] == "selected" and
                    abs(r["point_delta_better_minus_selected"]) >=
                    abs(r["width_delta_better_minus_selected"]))
    summary = {
        "n_pairs": n,
        "ucl_prefers_selected_count": count("ucl_prefers", "selected"),
        "ucl_prefers_better_count": count("ucl_prefers", "better"),
        "point_prefers_selected_count": count("point_prefers", "selected"),
        "width_prefers_selected_count": count("width_prefers", "selected"),
        "point_driven_width_offsets_count": count("point_width_class", "point_driven_width_offsets"),
        "mixed_point_dominant_count": count("point_width_class", "mixed_point_dominant"),
        "mixed_width_dominant_count": count("point_width_class", "mixed_width_dominant"),
        "uncertainty_driven_count": count("point_width_class", "uncertainty_driven"),
        "point_dominant_count": point_dom,
        "point_dominant_fraction": point_dom / n if n else None,
        "mean_point_delta": mean([r["point_delta_better_minus_selected"] for r in rows]) if rows else None,
        "mean_width_delta": mean([r["width_delta_better_minus_selected"] for r in rows]) if rows else None,
        "mean_ucl_delta": mean([r["ucl_delta_better_minus_selected"] for r in rows]) if rows else None,
        "mean_signed_point_fraction_of_ucl_margin": mean(
            [r["signed_point_fraction_of_ucl_margin"] for r in rows]) if rows else None,
        "mean_signed_width_fraction_of_ucl_margin": mean(
            [r["signed_width_fraction_of_ucl_margin"] for r in rows]) if rows else None,
    }
    return {"rows": rows, "summary": summary}
