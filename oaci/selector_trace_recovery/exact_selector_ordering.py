"""C37 selected-vs-better exact UCL comparisons."""
from __future__ import annotations

from . import artifact_loader as al
from . import leakage_ucl_replay
from . import schema


def _pref(delta, eps=schema.UCL_CLEAR_EPS):
    if not al.finite(delta):
        return "unavailable"
    if float(delta) > eps:
        return "selected"
    if float(delta) < -eps:
        return "better"
    return "flat"


def build_exact_comparisons(pairs, trace, recovered):
    rec = {r["pair_key"]: r for r in recovered["rows"]}
    ctx_cache = al.ContextCache(trace)
    rows = []
    for p in pairs:
        key = "|".join([p["seed"], p["target"], p["level"], p["selected_order"], p["candidate_order"]])
        rr = rec.get(key, {})
        ctx = ctx_cache.get(p["seed"], p["target"], p["level"], p["regime"])
        selected = leakage_ucl_replay.persisted_selected_leakage(ctx)
        srow = trace["by_key"][(p["seed"], p["target"], p["level"], p["regime"], p["selected_order"])]
        brow = trace["by_key"][(p["seed"], p["target"], p["level"], p["regime"], p["candidate_order"])]
        point_delta = al.as_float(brow.get("selection_leakage_point")) - al.as_float(srow.get("selection_leakage_point"))
        ucl_delta = al.as_float(rr.get("better_ucl")) - selected["bootstrap_ucl"]
        point_pref = _pref(point_delta)
        ucl_pref = _pref(ucl_delta)
        rows.append({
            "pair_id": p["pair_id"],
            "pair_key": key,
            "seed": p["seed"],
            "target": p["target"],
            "level": p["level"],
            "regime": p["regime"],
            "selected_order": p["selected_order"],
            "better_order": p["candidate_order"],
            "selected_candidate_id": srow["candidate_id"],
            "better_candidate_id": brow["candidate_id"],
            "selected_point": selected["extractable_LQ_ov"],
            "better_point": rr.get("better_point"),
            "point_delta_better_minus_selected": point_delta,
            "point_prefers": point_pref,
            "selected_ucl": selected["bootstrap_ucl"],
            "better_ucl": rr.get("better_ucl"),
            "ucl_delta_better_minus_selected": ucl_delta,
            "ucl_prefers": ucl_pref,
            "ucl_margin_abs": abs(ucl_delta) if al.finite(ucl_delta) else "",
            "pairwise_exact_selector_winner": ucl_pref,
            "rank_scope": "selected_vs_c35_preference_robust_better",
            "point_ucl_disagreement": int(point_pref in ("selected", "better") and ucl_pref in ("selected", "better") and
                                          point_pref != ucl_pref),
            "target_endpoint_prefers": "better",
            "fraction_weights_alt_beats_selected": p.get("fraction_weights_alt_beats_selected"),
            "utility_cone_category": p.get("utility_cone_category"),
            "recovery_status": rr.get("recovery_status", "missing"),
        })
    return rows


def summary(rows):
    n = len(rows)
    count = lambda v: sum(1 for r in rows if r["ucl_prefers"] == v)
    return {
        "n_pairs": n,
        "ucl_prefers_selected_count": count("selected"),
        "ucl_prefers_better_count": count("better"),
        "ucl_flat_count": count("flat"),
        "ucl_unavailable_count": count("unavailable"),
        "ucl_prefers_selected_fraction": count("selected") / n if n else None,
        "ucl_prefers_better_fraction": count("better") / n if n else None,
        "ucl_flat_fraction": count("flat") / n if n else None,
        "point_ucl_disagreement_fraction": (
            sum(r["point_ucl_disagreement"] for r in rows) / n if n else None),
    }

